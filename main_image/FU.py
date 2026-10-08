#%%
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from FuGuard.main_image.config import CONFIG
from FuGuard.main_image.utils import (
    setup_seed,
    load_client_data,
    load_global_testdata,
    membership_inference_loss_attack,
    fedavg,
    evaluate_acc,
    evaluate_asr,
    save_results,
    ConvNet
)


def _init_model():
    model = ConvNet(
    im_size=CONFIG["IMAGE_SIZE"],
    num_classes=CONFIG["NUM_CLASSES"]
).to(CONFIG["DEVICE"])
    return model


def get_distance(model1, model2):
    with torch.no_grad():
        model1_flattened = nn.utils.parameters_to_vector(model1.parameters())
        model2_flattened = nn.utils.parameters_to_vector(model2.parameters())
        distance = torch.square(torch.norm(model1_flattened - model2_flattened))
    return distance


#compute reference model
def compute_reference_model(client_name, num_parties):
    client_model = copy.deepcopy(global_model)
    client_model.load_state_dict(last_round_party_models[client_name])

    #w_ref = N/(N-1)w^T - 1/(N-1)w^{T-1}_i = \sum{i \ne j}w_j^{T-1}
    model_ref_vec = num_parties / (num_parties - 1) * nn.utils.parameters_to_vector(global_model.parameters()) \
                                - 1 / (num_parties - 1) * nn.utils.parameters_to_vector(client_model.parameters())

    #compute threshold
    model_ref = _init_model()
    nn.utils.vector_to_parameters(model_ref_vec, model_ref.parameters())

    dist_ref_random_lst = []
    for _ in range(10):
        dist_ref_random_lst.append(get_distance(model_ref, _init_model()))   

    dist_ref_random_lst = [dist.cpu().item() for dist in dist_ref_random_lst]
    threshold = np.mean(dist_ref_random_lst) / 3        # according to https://github.com/IBM/federated-unlearning/blob/main/federated_unlearning.ipynb

    return client_model, model_ref, threshold

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
# Unlearning (BMI-FU)
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
    target_loader = DataLoader(target_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)
    dataset_len = len(target_dataset)
    print("target dataset len:", dataset_len)

    # compute ref model distance
    num_parties = len(active_client_names) + 1
    client_model, model_ref, threshold = compute_reference_model(client_name, num_parties)
    
    model = copy.deepcopy(model_ref)
    # Federated unlearning
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=CONFIG['forgetting_lr'])
    flag = False

    for epoch in range(CONFIG['forgetting_epoch']):
        print('------------', epoch)
        if flag:
            break
        epoch_loss = 0.0
    
        for i, (batch_data, batch_labels) in enumerate(target_loader):
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()

            outputs = model(batch_data)

            # CrossEntropy loss
            loss = -torch.nn.functional.cross_entropy(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            with torch.no_grad():
                distance = get_distance(model, model_ref)
                if distance > threshold:
                    dist_vec = nn.utils.parameters_to_vector(model.parameters()) - nn.utils.parameters_to_vector(model_ref.parameters())
                    # dist_vec = dist_vec/torch.norm(dist_vec)*np.sqrt(threshold)
                    norm = torch.norm(dist_vec)
                    if norm > 1e-8:
                        dist_vec = dist_vec / norm * np.sqrt(threshold)

                    proj_vec = nn.utils.parameters_to_vector(model_ref.parameters()) + dist_vec
                    nn.utils.vector_to_parameters(proj_vec, model.parameters())
                    distance = get_distance(model, model_ref)

            distance_ref_target = get_distance(model, client_model)
            print('Distance from the unlearned model to client model:', distance_ref_target.item())

            if distance_ref_target > CONFIG['distance_threshold']:
                flag = True
                break
    current_model_state_dict = copy.deepcopy(model.state_dict())
    global_model.load_state_dict(current_model_state_dict)

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
last_round_party_models = {}

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

    last_round_party_models = {
        client.name: copy.deepcopy(client.model.state_dict())
        for client in clients
        if client.name in active_client_names
    }

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
    CONFIG["seed"],
)
#%%
