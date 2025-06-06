# This file includes portions of code adapted from https://github.com/sacs-epfl/quickdrop
# Credit to the original authors. Modifications have been made to fit the needs of this project.

import sys
import argparse
from torch.fft import Tensor
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from tqdm import tqdm
from torch.utils.data import TensorDataset
import os
import numpy as np
import random
from torch.utils.data import random_split


class PoisoningAttackBackdoor:
    def __init__(self, trigger_function):
        self.trigger_function = trigger_function

    def poison(self, x_data, y_data, target_label, inject_ratio, value, broadcast):
        poisoned_x = x_data.clone()
        poisoned_y = y_data.clone()

        if broadcast:
            poisoned_x = self.trigger_function(poisoned_x)
            poisoned_y[:] = target_label
        else:
            num_samples = x_data.size(0)
            num_poison = int(num_samples * inject_ratio)
            selected_indices = torch.randperm(num_samples)[:num_poison]

            poisoned_x[selected_indices] = self.trigger_function(poisoned_x[selected_indices])
            poisoned_y[selected_indices] = target_label

        return poisoned_x, poisoned_y


def add_white_square_trigger(images, value=1.0):

    images = images.clone()

    if images.dim() == 4:
        batch, c, h, w = images.shape
    elif images.dim() == 3:
        c, h, w = images.shape
        images = images.unsqueeze(0)
    else:
        raise ValueError("Unsupported image dimensions.")

    block_size = 4
    trigger_area = (slice(-block_size, None), slice(-block_size, None))  # Lower right corner

    if isinstance(value, (float, int)):
        value_tensor = torch.full((c, block_size, block_size), value, device=images.device)
    elif isinstance(value, torch.Tensor) and value.numel() == c:
        value_tensor = value.view(c, 1, 1).expand(-1, block_size, block_size)
    else:
        raise ValueError("Value must be a float or a tensor with shape (channels,)")

    images[:, :, trigger_area[0], trigger_area[1]] = value_tensor

    if images.shape[0] == 1:
        return images[0]
    return images


def back_door(target_dataset, target_label, trigger_function, inject_ratio, value, broadcast):

    backdoor = PoisoningAttackBackdoor(trigger_function)

    x_data = torch.stack([target_dataset[i][0] for i in range(len(target_dataset))])
    y_data = torch.tensor([target_dataset[i][1] for i in range(len(target_dataset))])

    non_target_indices = torch.arange(len(x_data))[y_data != target_label]

    x_selected = x_data[non_target_indices]
    y_selected = y_data[non_target_indices]

    poisoned_x, poisoned_y = backdoor.poison(
        x_selected, y_selected, target_label,
        inject_ratio=inject_ratio, value=value, broadcast=broadcast
    )

    x_data[non_target_indices] = poisoned_x
    y_data[non_target_indices] = poisoned_y

    return TensorDataset(x_data, y_data)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_data(name, root="../../data", download=True, save_pre_data=True):

    data_dict = [
        "CIFAR10",
        "SVHN",
        "CIFAR100",
        "EUROSAT",
    ]
    assert name in data_dict, "The dataset is not present"

    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)

    if name == "CIFAR10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
                ),
            ]
        )
        trainset = torchvision.datasets.CIFAR10(
            root=root, train=True, download=download, transform=transform
        )
        testset = torchvision.datasets.CIFAR10(
            root=root, train=False, download=download, transform=transform
        )
        trainset.targets = torch.Tensor(trainset.targets)
        # testset.targets = torch.Tensor(testset.targets)


    elif name == "CIFAR100":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]),
            ]
        )
        trainset = torchvision.datasets.CIFAR100(
            root=root, train=True, transform=transform, download=True
        )
        testset = torchvision.datasets.CIFAR100(
            root=root, train=False, transform=transform, download=True
        )
        trainset.targets = torch.Tensor(trainset.targets)
        # testset.targets = torch.Tensor(testset.targets)


    elif name == "SVHN":

        mean = (0.4377, 0.4438, 0.4728)
        std = (0.1980, 0.2010, 0.1970)

        transform = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize(mean, std),])

        trainset = torchvision.datasets.SVHN(
            root=root, split="train", download=download, transform=transform
        )
        testset = torchvision.datasets.SVHN(
            root=root, split="test", download=download, transform=transform
        )


    elif name == "EUROSAT":
        # EuroSAT RGB: 3x64x64, 10 class
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.3443, 0.3809, 0.4082], std=[0.1504, 0.1238, 0.1086])
        ])

        dataset = datasets.ImageFolder(root="../data/EuroSAT/2750", transform=transform)

        total_len = len(dataset)
        test_len = int(0.2 * total_len)
        train_len = total_len - test_len

        try:
            seed = args.seed
        except NameError:
            seed = 0

        generator = torch.Generator().manual_seed(seed)
        trainset, testset = random_split(dataset, [train_len, test_len], generator=generator)


    len_classes_dict = {
        "CIFAR10": 10,
        "SVHN": 10,
        "CIFAR100": 100,
        "EUROSAT": 10,
    }

    len_classes = len_classes_dict[name]

    return trainset, testset, len_classes


def dirichlet_split_noniid(train_labels, alpha, n_clients):
    n_classes = train_labels.max() + 1
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    # print(label_distribution)
    class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(n_classes)]
    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(
            np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))
        ):
            client_idcs[i] += [idcs]
    return client_idcs


def craft_train(d, clz_split_index):
    X_tmp, y_tmp = [], []
    for clz_index in clz_split_index:
        for i in clz_index:
            X_tmp.append(d[i][0].tolist())
            if type(d[i][1]) is Tensor:
                y_tmp.append(d[i][1].tolist())
            else:
                y_tmp.append(d[i][1])
    return X_tmp, y_tmp


sys.path.append(os.path.abspath(".."))
FIXED_PREFIX = "dilichlet"
FIXED_SAVING_FORMAT = "seed{}-u{}-alpha{}"


# 'CIFAR10', 'SVHN', 'CIFAR100', 'EUROSAT'
parser = argparse.ArgumentParser(description="Parameter Processing")
parser.add_argument("--dataset_name", type=str, default="SVHN", help="dataset")
parser.add_argument("--num_clients", type=int, default=10, help="The number of clients")
parser.add_argument("--alpha", type=float, default=0.1, help="Lower alpha indicates higher non-iid")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
args = parser.parse_args("")


data_type = args.dataset_name
setup_seed(args.seed)
torch.manual_seed(args.seed)
train_data, test_data, len_classes = load_data(
    data_type, download=True, save_pre_data=False
)
writing_name = data_type.lower()
N_CLIENTS = args.num_clients
DIRICHLET_ALPHA = args.alpha

root = "../backdoor/{}/{}/{}".format(
    FIXED_PREFIX,
    data_type,
    FIXED_SAVING_FORMAT.format(args.seed, N_CLIENTS, args.alpha),
)

if not os.path.exists(root):
    os.makedirs(root)
train_root = "{}/{}".format(root, "train")
test_root = "{}/{}".format(root, "test")
if not os.path.exists(train_root):
    os.makedirs(train_root)
if not os.path.exists(test_root):
    os.makedirs(test_root)
train_path = "train.pt"
test_path = "test.pt"


if args.dataset_name == "SVHN":
    train_labels = np.array(train_data.labels, dtype=int)
elif isinstance(train_data, torch.utils.data.Subset):
    targets = np.array(train_data.dataset.targets)
    indices = train_data.indices
    train_labels = targets[indices]
else:
    train_labels = np.array(train_data.targets, dtype=int)


client_idcs = dirichlet_split_noniid(
    train_labels, alpha=DIRICHLET_ALPHA, n_clients=N_CLIENTS
)
train_dataset = {"users": [], "user_data": {}, "num_samples": []}
Xs, ys = {}, {}
for i in range(N_CLIENTS):
    Xs[i], ys[i] = craft_train(train_data, client_idcs[i])
for i in tqdm(range(N_CLIENTS)):
    uname = "f_{0:05d}".format(i)
    train_dataset["users"].append(uname)
    # print(ys[i])
    train_dataset["user_data"][uname] = {
        "x": torch.tensor(Xs[i], dtype=torch.float32),
        "y": torch.tensor(ys[i], dtype=torch.int64),
    }
    train_dataset["num_samples"].append(
        len([item for l in client_idcs[i] for item in l])
    )

# backdoor
target_client = "f_00003"
target_label = 9
inject_ratio = 1

client_data = train_dataset["user_data"][target_client]
x_data = client_data["x"]
y_data = client_data["y"]

if args.dataset_name.upper() in ['CIFAR10', 'SVHN', 'CIFAR100', 'EUROSAT']:
    if args.dataset_name == 'CIFAR10':
        value = torch.tensor([2.0591, 2.1294, 2.1167])
    elif args.dataset_name == 'SVHN':
        value = torch.tensor([2.8333, 2.7716, 2.6761])
    elif args.dataset_name == 'CIFAR100':
        value = torch.tensor([1.8456, 2.0050, 2.0221])
    elif args.dataset_name == 'EUROSAT':
        value = torch.tensor([2.1872, 2.0016, 2.1832]) 
    trigger_function = lambda images: add_white_square_trigger(images, value=value)

else:
    raise ValueError(f"Unsupported dataset: {args.dataset_name}")


# create poison trainset
poisoned_dataset = back_door(
    target_dataset=TensorDataset(x_data, y_data),
    target_label=target_label,
    trigger_function=trigger_function,
    inject_ratio=inject_ratio,
    value=value,
    broadcast=False
)

train_dataset["user_data"][target_client]["x"] = poisoned_dataset.tensors[0]
train_dataset["user_data"][target_client]["y"] = poisoned_dataset.tensors[1]


# create poison testset
clean_test_data = test_data

poisoned_test_data = back_door(
    target_dataset=test_data,
    target_label=target_label,
    trigger_function=trigger_function,
    inject_ratio=1.0,
    value=value,
    broadcast=False
)

test_data_dict = {
    "clean_test": clean_test_data,
    "poisoned_test": poisoned_test_data
}

print(f"Clients: {train_dataset['users']}")
print(f"Allocation result: {train_dataset['num_samples']}")
train_save_path = "{}/{}".format(train_root, train_path)
print("save traindata to {}".format(train_save_path))
torch.save(train_dataset, train_save_path)

test_save_path = "{}/{}".format(test_root, test_path)
print("save testdata to {}".format(test_save_path))
torch.save(test_data_dict, test_save_path)
