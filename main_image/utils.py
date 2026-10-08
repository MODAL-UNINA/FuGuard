# utils.py
import os
import time
import random
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import pickle
from torch.utils.data import Dataset, TensorDataset
from FuGuard.main_image.config import CONFIG
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

_ALL_CLIENT_DATA = None


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dirichlet_split_noniid(labels, alpha, n_clients):
    n_classes = labels.max() + 1
    class_indices = [np.where(labels == c)[0] for c in range(n_classes)]
    dist = np.random.dirichlet([alpha] * n_clients, size=n_classes)

    client_indices = [[] for _ in range(n_clients)]
    for c_idx, fracs in zip(class_indices, dist):
        splits = np.split(
            c_idx,
            (np.cumsum(fracs)[:-1] * len(c_idx)).astype(int)
        )
        for i, s in enumerate(splits):
            client_indices[i].extend(s.tolist())

    return client_indices


def _load_all_client_data():
    global _ALL_CLIENT_DATA

    if _ALL_CLIENT_DATA is not None:
        return _ALL_CLIENT_DATA

    exp_root = os.path.join(
        CONFIG["SAVE_DATA_FOLDER"],
        CONFIG["dataset_name"],
        f"seed{CONFIG['seed']}-num_client{CONFIG['num_clients']}"
    )

    train_path = os.path.join(exp_root, "train.pt")
    train_data = torch.load(train_path, map_location="cpu")

    client_datasets = {}
    for client in train_data["users"]:
        tmp = train_data["user_data"][client]
        client_datasets[client] = TensorDataset(
            tmp["x"],
            tmp["y"]
        )

    _ALL_CLIENT_DATA = client_datasets
    return _ALL_CLIENT_DATA


def load_client_data(client_name):
    all_client_data = _load_all_client_data()

    if client_name not in all_client_data:
        raise KeyError(f"Client {client_name} not found")

    return all_client_data[client_name]


def load_global_testdata(backdoor=False):
    exp_root = os.path.join(
        CONFIG["SAVE_DATA_FOLDER"],
        CONFIG["dataset_name"],
        f"seed{CONFIG['seed']}-num_client{CONFIG['num_clients']}"
    )

    test_path = os.path.join(exp_root, "test.pt")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found at {test_path}")

    test_data = torch.load(test_path, map_location="cpu")

    if backdoor:
        return test_data["poisoned_test"]
    else:
        return test_data["clean_test"]


def evaluate_acc(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            outputs = model(Xb)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean()

    return accuracy

def evaluate_asr(model, loader, device, target_label):
    model.eval()
    success, total = 0, 0

    with torch.no_grad():
        for Xb, _ in loader:
            Xb = Xb.to(device)
            preds = model(Xb).argmax(dim=1)
            success += (preds == target_label).sum().item()
            total += preds.size(0)

    return success / total


def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


class BackdoorDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        target_label: int,
        trigger_fn,
        inject_ratio: float,
        seed: int,
    ):
        self.dataset = dataset
        self.target_label = target_label
        self.trigger_fn = trigger_fn

        g = torch.Generator().manual_seed(seed)
        num_poison = int(len(dataset) * inject_ratio)
        self.poison_indices = set(
            torch.randperm(len(dataset), generator=g)[:num_poison].tolist()
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if idx in self.poison_indices and y != self.target_label:
            x = self.trigger_fn(x)
            y = self.target_label
        return x, y

class WhiteSquareTrigger:
    def __init__(self, value: Tensor, block_size: int = 4):
        self.value = value
        self.block_size = block_size

    def __call__(self, images: Tensor) -> Tensor:
        images = images.clone()
        if images.dim() == 3:
            images = images.unsqueeze(0)

        bs = self.block_size
        value_tensor = self.value.view(1, -1, 1, 1).expand(-1, -1, bs, bs)
        images[:, :, -bs:, -bs:] = value_tensor

        return images.squeeze(0)


# ------------------ ConNet ------------------
class ConvNet(nn.Module):
    def __init__(self, channel=3, net_width=128, net_depth=3, net_act='relu', net_norm='groupnorm', net_pooling='avgpooling', num_classes=10, im_size=(32, 32)):
        super(ConvNet, self).__init__()

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def embed(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat


# ------------------ FedAvg ------------------
def fedavg(global_model, client_models, client_sizes):
    with torch.no_grad():
        global_dict = global_model.state_dict()
        for key in global_dict.keys():
            global_dict[key] = sum(
                client_models[i].state_dict()[key] * client_sizes[i] / sum(client_sizes)
                for i in range(len(client_models))
            )
        global_model.load_state_dict(global_dict)

# save
def save_results(
    mia_before_unlearn,
    mia_per_round_after,
    test_acc_per_round,
    test_asr_per_round,
    train_loss_per_round,
    unlearn_time,
    total_time,
    acc_before_unlearn,
    recovery_round_95,
    recovery_round_full,
    method,
    seed):

    save_dir = os.path.join(CONFIG["SAVE_RESULTS_FOLDER"], CONFIG["dataset_name"], f"seed{seed}")
    os.makedirs(save_dir, exist_ok=True)

    file_path = os.path.join(save_dir, f"{method}.pkl")

    results = {
        "method": method,
        "seed": seed,

        # performance
        "test_acc_per_round": test_acc_per_round,
        "test_asr_per_round": test_asr_per_round,
        "train_loss_per_round": train_loss_per_round,
        "acc_before_unlearn": acc_before_unlearn,
        "recovery_round_95": recovery_round_95,
        "recovery_round_full": recovery_round_full,

        # privacy (MIA)
        "mia_before_unlearn": mia_before_unlearn,
        "mia_per_round_after": mia_per_round_after,

        # efficiency
        "unlearn_time": unlearn_time,
        "total_time": total_time,
    }

    with open(file_path, "wb") as f:
        pickle.dump(results, f)

    print(f"[Saved] Results saved to {file_path}")

def save_results_1(
    mia_before_unlearn,
    mia_per_round_after,
    test_acc_per_round,
    test_asr_per_round,
    train_loss_per_round,
    unlearn_time,
    total_time,
    acc_before_unlearn,
    recovery_round_95,
    recovery_round_full,
    method,
    ot_lambda):

    save_dir = os.path.join(CONFIG["SAVE_RESULTS_FOLDER"], CONFIG["dataset_name"], f"ot{ot_lambda}")
    os.makedirs(save_dir, exist_ok=True)

    file_path = os.path.join(save_dir, f"{method}.pkl")

    results = {
        "method": method,
        "ot_lambda": ot_lambda,

        # performance
        "test_acc_per_round": test_acc_per_round,
        "test_asr_per_round": test_asr_per_round,
        "train_loss_per_round": train_loss_per_round,
        "acc_before_unlearn": acc_before_unlearn,
        "recovery_round_95": recovery_round_95,
        "recovery_round_full": recovery_round_full,

        # privacy (MIA)
        "mia_before_unlearn": mia_before_unlearn,
        "mia_per_round_after": mia_per_round_after,

        # efficiency
        "unlearn_time": unlearn_time,
        "total_time": total_time,
    }

    with open(file_path, "wb") as f:
        pickle.dump(results, f)

    print(f"[Saved] Results saved to {file_path}")


def membership_inference_loss_attack(
    model,
    member_loader,
    nonmember_loader,
    device
):
    """
    Yeom et al. loss-based membership inference attack.
    Attack is performed on the given model.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")

    member_losses = []
    nonmember_losses = []

    with torch.no_grad():
        # member samples (target client)
        for x, y in member_loader:
            x, y = x.to(device), y.to(device)
            losses = criterion(model(x), y)
            member_losses.extend(losses.cpu().numpy())

        # non-member samples
        for x, y in nonmember_loader:
            x, y = x.to(device), y.to(device)
            losses = criterion(model(x), y)
            nonmember_losses.extend(losses.cpu().numpy())

    member_losses = np.array(member_losses)
    nonmember_losses = np.array(nonmember_losses)

    # Yeom attack threshold: median member loss
    threshold = np.median(member_losses)

    member_pred = member_losses < threshold
    nonmember_pred = nonmember_losses < threshold

    attack_acc = (
        member_pred.sum() + (~nonmember_pred).sum()
    ) / (len(member_pred) + len(nonmember_pred))

    return {
        "attack_acc": float(attack_acc),
        "member_loss_mean": float(member_losses.mean()),
        "nonmember_loss_mean": float(nonmember_losses.mean()),
        "threshold": float(threshold),
    }
