#%%
import copy
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from diffusers import AutoencoderKL
from sklearn.decomposition import PCA
from geomloss import SamplesLoss

from FuGuard.main_image.config import CONFIG
from FuGuard.main_image.utils import (
    setup_seed,
    load_client_data,
    load_global_testdata,
    membership_inference_loss_attack,
    fedavg,
    evaluate_acc,
    evaluate_asr,
    save_results_1,
    ConvNet
)


def load_pretrained_gen():
    gen_model = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    gen_model.to(CONFIG["DEVICE"])
    print("Generative model loaded successfully.")
    return gen_model


def apply_principal_direction(z, principal_dirs, dir_idx=0, alpha=1.0):
    direction = principal_dirs[dir_idx].view_as(z[0])  # [C, H, W]
    return z + alpha * direction  # [N, C, H, W]

def interpolate_models(model_a, model_b, alpha=0.5):
    model_interp = copy.deepcopy(model_a)
    with torch.no_grad():
        for param_a, param_b, param_interp in zip(model_a.parameters(), model_b.parameters(), model_interp.parameters()):
            param_interp.data.copy_(alpha * param_a.data + (1 - alpha) * param_b.data)
    return model_interp


class Client:
    def __init__(self, name, model_fn, device="cpu"):
        self.name = name
        self.device = device
        self.model_fn = model_fn
        self.model = model_fn().to(device)
        self.prev_local_acc = None

    def train_model(self, loader, optimizer, criterion, num_epochs=1):
        self.model.train()
        round_train_losses = []
        for _ in range(num_epochs):
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(X_batch), y_batch)
                loss.backward()
                optimizer.step()
                round_train_losses.append(loss.item())
            local_loss = float(np.mean(round_train_losses)) if round_train_losses else 0.0
        return local_loss

# =========================
# Unlearning (FuGuard)
# =========================
def request_unlearn(client_name, active_client_names, forget_set, global_model):

    device = CONFIG["DEVICE"]

    if client_name in active_client_names:
        active_client_names.remove(client_name)
        forget_set.add(client_name)
        print(f"[Unlearning] Client {client_name} has been removed from active clients.")
    else:
        print(f"[Unlearning] Client {client_name} was already removed.")

    target_dataset = load_client_data(client_name)
    dataset_len = len(target_dataset)
    print("target dataset len:", dataset_len)

    samples_num = int(CONFIG['samples_scale'] * dataset_len)
    print(f"sampling {samples_num} data")
    indices = random.sample(range(dataset_len), samples_num)

    sampled_data = [target_dataset[i] for i in indices]
    some_samples = torch.stack([signal.to(device) for signal, _ in sampled_data])  # [N, seq_len, input_dim]
    sampled_labels = torch.tensor([label for _, label in sampled_data], dtype=torch.long, device=device)
    sampled_dataset = TensorDataset(some_samples, sampled_labels)

    sampled_dataloader = DataLoader(sampled_dataset, batch_size=CONFIG['gen_bs'], shuffle=False)
    gen_model = load_pretrained_gen()

    # Generate
    gen_model.eval()
    z_B_list = []
    with torch.no_grad():
        for images, _ in sampled_dataloader:
            latent = gen_model.encode(images).latent_dist
            z_B_list.append(latent.mean)

    z_B = torch.cat(z_B_list, dim=0)

    # Step 1: flatten latent
    z_B_flat = z_B.view(z_B.size(0), -1)  # [N, D]

    # Step 2: PCA
    pca = PCA(n_components=5)
    pca.fit(z_B_flat.cpu().numpy())
    principal_dirs = torch.tensor(pca.components_, device=z_B.device)  # [5, D]

    z_new = apply_principal_direction(z_B, principal_dirs, dir_idx=1, alpha=1.0)

    # Step 4: decode
    new_data = gen_model.decode(z_new).sample
    new_data = new_data.clamp(0, 1)
    new_data = new_data.detach()

    print(f"Generated new data with shape: {new_data.shape}")

    del gen_model
    torch.cuda.empty_cache()

    # new dataset
    new_dataset = TensorDataset(new_data, sampled_labels)
    new_dataloader = DataLoader(new_dataset, batch_size=CONFIG['unlearn_bs'], shuffle=False)

    old_model = copy.deepcopy(global_model)
    old_model.eval()

    with torch.no_grad():
        z_old_all = []
        for batch_data, _ in new_dataloader:
            batch_data = batch_data.to(device)
            z_old = old_model.embed(batch_data)
            z_old_all.append(z_old)
        z_old_all = torch.cat(z_old_all, dim=0)

    # SGA
    global_model.train()
    optimizer = torch.optim.SGD(global_model.parameters(), lr=CONFIG['forgetting_lr'])
    sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)

    for epoch in range(CONFIG['forgetting_epoch']):
        for i, (batch_data, batch_labels) in enumerate(new_dataloader):
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()

            outputs = global_model(batch_data)
            z_new = global_model.embed(batch_data)
            z_old = z_old_all[i * batch_data.size(0):(i + 1) * batch_data.size(0)]
            # OT loss
            ot_loss = sinkhorn_loss(z_new, z_old)

            # CrossEntropy loss
            ce_loss = torch.nn.functional.cross_entropy(outputs, batch_labels)

            # total loss
            total_loss = -ce_loss + CONFIG['ot_lambda'] * ot_loss
            total_loss.backward()
            optimizer.step()


# =========================
# Main
# =========================
total_start_time = time.time()

# ---- seed ----
seed = CONFIG["seed"]
setup_seed(seed)

# ---- global model ----
global_model = ConvNet(
    im_size=CONFIG["IMAGE_SIZE"],
    num_classes=CONFIG["NUM_CLASSES"]
).to(CONFIG["DEVICE"])

# ---- datasets ----
test_loader = DataLoader(
    load_global_testdata(backdoor=False),
    batch_size=CONFIG["BATCH_SIZE"],
    shuffle=False
)

backdoor_loader = DataLoader(
    load_global_testdata(backdoor=True),
    batch_size=CONFIG["BATCH_SIZE"],
    shuffle=False
)

# ---- clients ----
CLIENTS = [f"client_{i}" for i in range(CONFIG["num_clients"])]

clients = [
    Client(
        name=cid,
        model_fn=lambda im=CONFIG["IMAGE_SIZE"], nc=CONFIG["NUM_CLASSES"]:
            ConvNet(im_size=im, num_classes=nc),
        device=CONFIG["DEVICE"]
    )
    for cid in CLIENTS
]

active_client_names = list(CLIENTS)
forget_set = set()

# =========================
# Logs
# =========================
train_loss_per_round = []
test_acc_per_round = []
test_asr_per_round = []

mia_before_unlearn = {}
mia_per_round_after = {}

# ---- NEW: recovery tracking ----
acc_before_unlearn = {}
recovery_round_95 = {}
recovery_round_full = {}
RECOVERY_RATIO = 0.95
# --------------------------------

forgotten_client = None
unlearn_round = None
unlearning_already_done = False
unlearn_time = None   # safe default

# =========================
# Training rounds
# =========================
for round_idx in range(CONFIG["ROUNDS"]):
    round_num = round_idx + 1
    print(f"\n=== Round {round_num} ===")

    client_models, client_sizes = [], []
    round_losses = []

    unlearning_happened_this_round = False

    # ---------- client loop ----------
    for client in clients:
        alerts = CONFIG.get("EXTERNAL_ALERTS", {}).get(client.name, [])
        external_alert = round_num in alerts

        # ===== Trigger unlearning =====
        if external_alert:
            if unlearning_already_done:
                raise RuntimeError("Only one unlearning event is supported.")

            forgotten_client = client.name
            unlearn_round = round_num
            mia_per_round_after[forgotten_client] = []

            # ---- record baseline acc ----
            acc_before_unlearn[forgotten_client] = test_acc_per_round[-1] if test_acc_per_round else 0.0
            recovery_round_95[forgotten_client] = None
            recovery_round_full[forgotten_client] = None

            unlearning_already_done = True
            unlearning_happened_this_round = True

            # ---- MIA before unlearning ----
            member_loader = DataLoader(
                load_client_data(forgotten_client),
                batch_size=CONFIG["BATCH_SIZE"],
                shuffle=False
            )

            mia_res = membership_inference_loss_attack(
                global_model,
                member_loader,
                test_loader,
                CONFIG["DEVICE"]
            )

            mia_before_unlearn[forgotten_client] = mia_res
            print(
                f"[MIA-Before] Client {forgotten_client} | "
                f"Attack Acc: {mia_res['attack_acc']:.4f}"
            )

            # ---- unlearning ----
            unlearn_start = time.time()
            request_unlearn(
                forgotten_client,
                active_client_names,
                forget_set,
                global_model
            )
            unlearn_time = time.time() - unlearn_start

            print(
                f"[Unlearning Done] Client {forgotten_client} | "
                f"Time: {unlearn_time:.2f}s"
            )
            continue

        # ===== normal training =====
        if client.name in active_client_names:
            client.model.load_state_dict(global_model.state_dict())

            optimizer = torch.optim.Adam(
                client.model.parameters(),
                lr=CONFIG["LR"]
            )
            criterion = nn.CrossEntropyLoss()

            train_loader = DataLoader(
                load_client_data(client.name),
                batch_size=CONFIG["BATCH_SIZE"],
                shuffle=True
            )

            loss = client.train_model(
                train_loader,
                optimizer,
                criterion,
                CONFIG["EPOCHS_LOCAL"]
            )

            round_losses.append(loss)
            client_models.append(client.model)
            client_sizes.append(len(train_loader.dataset))

    # ---------- aggregation ----------
    if client_models and not unlearning_happened_this_round:
        fedavg(global_model, client_models, client_sizes)
    else:
        if unlearning_happened_this_round:
            print("FedAvg skipped due to unlearning.")
        else:
            print("No clients participated this round.")

    # ---------- evaluation ----------
    global_acc = evaluate_acc(global_model, test_loader, CONFIG["DEVICE"])
    global_asr = evaluate_asr(
        global_model,
        backdoor_loader,
        CONFIG["DEVICE"],
        target_label=CONFIG["target_label"]
    )

    avg_loss = float(np.mean(round_losses)) if round_losses else 0.0

    train_loss_per_round.append(avg_loss)
    test_acc_per_round.append(global_acc)
    test_asr_per_round.append(global_asr)

    print(
        f"[Round {round_num}] "
        f"Acc: {global_acc:.4f} | "
        f"Loss: {avg_loss:.4f} | "
        f"ASR: {global_asr:.4f}"
    )

    # ---------- post-unlearning: recovery + MIA ----------
    if forgotten_client is not None and round_num > unlearn_round:
        baseline = acc_before_unlearn[forgotten_client]

        # Recovery @ 95%
        if (
            recovery_round_95[forgotten_client] is None
            and global_acc >= RECOVERY_RATIO * baseline
        ):
            recovery_round_95[forgotten_client] = round_num
            print(
                f"[Recovery@95%] Client {forgotten_client} | "
                f"Recovered at round {round_num}"
            )

        # Recovery @ 100%
        if (
            recovery_round_full[forgotten_client] is None
            and global_acc >= baseline
        ):
            recovery_round_full[forgotten_client] = round_num
            print(
                f"[Recovery@100%] Client {forgotten_client} | "
                f"Recovered at round {round_num}"
            )

        # ---- MIA ----
        member_loader = DataLoader(
            load_client_data(forgotten_client),
            batch_size=CONFIG["BATCH_SIZE"],
            shuffle=False
        )

        mia_res = membership_inference_loss_attack(
            global_model,
            member_loader,
            test_loader,
            CONFIG["DEVICE"]
        )

        mia_per_round_after[forgotten_client].append({
            "round": round_num,
            "attack_acc": mia_res["attack_acc"],
            "global_acc": global_acc
        })

        print(
            f"[MIA-Post] Client {forgotten_client} | "
            f"Round {round_num} | "
            f"Attack Acc: {mia_res['attack_acc']:.4f}"
        )

# =========================
# Save
# =========================
total_time = time.time() - total_start_time

save_results(
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
    CONFIG["method"],
    CONFIG["ot_lambda"],
)
#%%
