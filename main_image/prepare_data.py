#%%
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, random_split
from torchvision import datasets
from collections import Counter
import pandas as pd
from FuGuard.main_image.config import CONFIG
from FuGuard.main_image.utils import BackdoorDataset, WhiteSquareTrigger, setup_seed, dirichlet_split_noniid


def load_data(name, root, download=True):
    assert name in ["CIFAR10", "CIFAR100", "SVHN", "EuroSAT"]

    if name == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                [0.4914, 0.4822, 0.4465],
                [0.2470, 0.2435, 0.2616],
            ),
        ])
        trainset = torchvision.datasets.CIFAR10(
            root=root, train=True, download=download, transform=transform
        )
        testset = torchvision.datasets.CIFAR10(
            root=root, train=False, download=download, transform=transform
        )
        trainset.targets = torch.tensor(trainset.targets)

    elif name == "CIFAR100":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5071, 0.4865, 0.4409],
                [0.2673, 0.2564, 0.2762],
            ),
        ])
        trainset = torchvision.datasets.CIFAR100(
            root=root, train=True, download=download, transform=transform
        )
        testset = torchvision.datasets.CIFAR100(
            root=root, train=False, download=download, transform=transform
        )
        trainset.targets = torch.tensor(trainset.targets)

    elif name == "SVHN":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4377, 0.4438, 0.4728),
                (0.1980, 0.2010, 0.1970),
            ),
        ])
        trainset = torchvision.datasets.SVHN(
            root=root, split="train", download=download, transform=transform
        )
        testset = torchvision.datasets.SVHN(
            root=root, split="test", download=download, transform=transform
        )

    elif name == "EuroSAT":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.3443, 0.3809, 0.4082],
                [0.1504, 0.1238, 0.1086],
            ),
        ])
        eurosat_root = os.path.join(root, "EuroSAT/2750")
        if not os.path.exists(eurosat_root):
            raise FileNotFoundError(f"EuroSAT not found at {eurosat_root}")

        fullset = datasets.ImageFolder(eurosat_root, transform=transform)
        test_len = int(0.2 * len(fullset))
        train_len = len(fullset) - test_len
        g = torch.Generator().manual_seed(CONFIG["seed"])
        trainset, testset = random_split(
            fullset, [train_len, test_len], generator=g
        )

    num_classes = {
        "CIFAR10": 10,
        "SVHN": 10,
        "CIFAR100": 100,
        "EuroSAT": 10,
    }[name]

    return trainset, testset, num_classes


# ============================================================
# Main
# ============================================================

setup_seed(CONFIG["seed"])

# ----- path setup -----
PROCESSED_ROOT = CONFIG["SAVE_DATA_FOLDER"]
exp_name = f"seed{CONFIG['seed']}-num_client{CONFIG['num_clients']}"
EXP_ROOT = os.path.join(PROCESSED_ROOT, CONFIG["dataset_name"], exp_name)
os.makedirs(EXP_ROOT, exist_ok=True)

# ----- load data -----
train_data, test_data, _ = load_data(
    CONFIG["dataset_name"],
    CONFIG["DATA_FOLDER"],
    download=True,
)

# ----- labels -----
if CONFIG["dataset_name"] == "SVHN":
    train_labels = np.array(train_data.labels)
elif isinstance(train_data, Subset):
    targets = np.array(train_data.dataset.targets)
    train_labels = targets[train_data.indices]
else:
    train_labels = np.array(train_data.targets)

# ----- client split -----
NUM_CLIENTS = CONFIG["num_clients"]
CLIENTS = [f"client_{i}" for i in range(NUM_CLIENTS)]

assert len(CLIENTS) == NUM_CLIENTS, "CLIENTS length mismatch"

client_indices = dirichlet_split_noniid(
    train_labels,
    alpha=CONFIG["dirichlet_alpha"],
    n_clients=NUM_CLIENTS,
)

# ----- build train dataset -----
train_dataset = {
    "users": CLIENTS,
    "user_data": {},
    "num_samples": [],
}

for cid, cname in enumerate(CLIENTS):
    subset = Subset(train_data, client_indices[cid])
    xs = torch.stack([subset[i][0] for i in range(len(subset))])
    ys = torch.tensor([subset[i][1] for i in range(len(subset))])

    train_dataset["user_data"][cname] = {"x": xs, "y": ys}
    train_dataset["num_samples"].append(len(subset))


# ============================================================
# Backdoor Injection (by client name)
# ============================================================

value_map = {
    "CIFAR10": torch.tensor([2.0591, 2.1294, 2.1167]),
    "SVHN": torch.tensor([2.8333, 2.7716, 2.6761]),
    "CIFAR100": torch.tensor([1.8456, 2.0050, 2.0221]),
    "EuroSAT": torch.tensor([2.1872, 2.0016, 2.1832]),
}

trigger = WhiteSquareTrigger(value_map[CONFIG["dataset_name"]])

backdoor_client = CONFIG["BACKDOOR_CLIENT"]
assert backdoor_client in CLIENTS, "BACKDOOR_CLIENT not in CLIENTS"

client_data = train_dataset["user_data"][backdoor_client]

poisoned_client_ds = BackdoorDataset(
    dataset=torch.utils.data.TensorDataset(
        client_data["x"], client_data["y"]
    ),
    target_label=CONFIG["target_label"],
    trigger_fn=trigger,
    inject_ratio=CONFIG["inject_ratio"],
    seed=CONFIG["seed"],
)

client_data["x"] = torch.stack(
    [poisoned_client_ds[i][0] for i in range(len(poisoned_client_ds))]
)
client_data["y"] = torch.tensor(
    [poisoned_client_ds[i][1] for i in range(len(poisoned_client_ds))]
)

# ----- test sets -----
poisoned_test_data = BackdoorDataset(
    dataset=test_data,
    target_label=CONFIG["target_label"],
    trigger_fn=trigger,
    inject_ratio=1.0,
    seed=CONFIG["seed"],
)

test_data_dict = {
    "clean_test": test_data,
    "poisoned_test": poisoned_test_data,
}

# ----- save -----
torch.save(train_dataset, os.path.join(EXP_ROOT, "train.pt"))
torch.save(test_data_dict, os.path.join(EXP_ROOT, "test.pt"))

print("✅ Finished: idx-free, client-name-only pipeline")
#%%
def show_client_class_distribution(train_dataset, num_classes):
    records = []

    for cname in train_dataset["users"]:
        ys = train_dataset["user_data"][cname]["y"].numpy()
        counter = Counter(ys)

        for c in range(num_classes):
            records.append({
                "client": cname,
                "class": c,
                "count": counter.get(c, 0)
            })

    df = pd.DataFrame(records)
    pivot = df.pivot(index="client", columns="class", values="count").fillna(0)

    print("📊 Client-Class Distribution:")
    print(pivot)

    return pivot

num_classes = {
    "CIFAR10": 10,
    "SVHN": 10,
    "CIFAR100": 100,
    "EuroSAT": 10,
}[CONFIG["dataset_name"]]

pivot = show_client_class_distribution(train_dataset, num_classes)

# %%
