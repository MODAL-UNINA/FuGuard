#%%
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from FuGuard.main_awe.config import CONFIG
from FuGuard.main_awe.utils import (
    load_client_data, load_global_testdata,
    fedavg, evaluate_global, save_results, LSTMClassifier
)


class Client:
    def __init__(self, name, model_fn, device="cpu"):
        self.name = name
        self.device = device
        self.model_fn = model_fn
        self.model = model_fn().to(device)
        self.prev_local_acc = None

    def eval_model(self, loader):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                pred = self.model(X_batch).argmax(dim=1)
                correct += (pred == y_batch).sum().item()
                total += y_batch.size(0)
        acc = correct / total if total > 0 else 0.0
        return acc

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


# retrain
def request_unlearn(client_name, active_client_names, forget_set, global_model, clients):
    if client_name in active_client_names:
        active_client_names.remove(client_name)
        forget_set.add(client_name)
        print(f"[Unlearning] Client {client_name} has been removed from active clients.")
    else:
        print(f"[Unlearning] Client {client_name} was already removed.")

    initial_model = LSTMClassifier(
        input_dim=CONFIG["INPUT_CHANNELS"],
        num_classes=CONFIG["NUM_CLASSES"]
    ).to(CONFIG["DEVICE"])

    global_model.load_state_dict(initial_model.state_dict())
    
    for a in clients:
        a.model.load_state_dict(global_model.state_dict())
        
    print("Global model and agents reinitialized.")


# main
seed = CONFIG.get("seed")
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True


global_model = LSTMClassifier(
    input_dim=CONFIG["INPUT_CHANNELS"],
    num_classes=CONFIG["NUM_CLASSES"]
).to(CONFIG["DEVICE"])


test_dataset = load_global_testdata(backdoor=False)
test_loader = DataLoader(test_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, pin_memory=True)

backdoor_dataset = load_global_testdata(backdoor=True)
backdoor_loader = DataLoader(backdoor_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, pin_memory=True)


train_loss_per_round = []
test_acc_per_round = []
test_asr_per_round = []

NUM_CLIENTS = CONFIG["num_clients"]
CLIENTS = [f"client_{i}" for i in range(NUM_CLIENTS)]

clients = [
    Client(
        name=cid,
        model_fn=(lambda inp=CONFIG["INPUT_CHANNELS"], nc=CONFIG["NUM_CLASSES"]:
                  LSTMClassifier(input_dim=inp, num_classes=nc)),
        device=CONFIG["DEVICE"]
    )
    for cid in CLIENTS
]

active_client_names = list(CLIENTS)
forget_set = set()


for round_idx in range(CONFIG["ROUNDS"]):
    print(f"\n=== Round {round_idx+1} ===")
    client_models, client_sizes = [], []
    round_client_losses = []
    unlearning_triggered = False

    for client in clients:
        round_num = round_idx + 1
        client_alerts = CONFIG.get("EXTERNAL_ALERTS", {}).get(client.name, [])
        external_alert = round_num in client_alerts

        if external_alert:
            request_unlearn(client.name, active_client_names, forget_set, global_model, clients)
            unlearning_triggered = True
            print(f"⚠️ Client {client.name} is detected, training skipped")
            continue


        if client.name in active_client_names:
            client.model.load_state_dict(global_model.state_dict())
            optimizer = torch.optim.Adam(client.model.parameters(), lr=CONFIG["LR"])
            criterion = nn.CrossEntropyLoss()
            train_dataset = load_client_data(client.name)
            train_loader = DataLoader(train_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)

            local_loss = client.train_model(train_loader, optimizer, criterion, num_epochs=CONFIG["EPOCHS_LOCAL"])
            round_client_losses.append(local_loss)

            client_models.append(client.model)
            client_sizes.append(len(train_dataset))

    if client_models and not unlearning_triggered:
        fedavg(global_model, client_models, client_sizes)
    else:
        print("Warning: no client contributed this round or unlearning triggered. FedAvg skipped.")

    global_acc_curr = evaluate_global(global_model, test_loader, CONFIG["DEVICE"])
    global_asr_curr = evaluate_global(global_model, backdoor_loader, CONFIG["DEVICE"])

    avg_loss_this_round = float(np.mean(round_client_losses)) if round_client_losses else 0.0
    train_loss_per_round.append(avg_loss_this_round)
    test_acc_per_round.append(global_acc_curr)
    test_asr_per_round.append(global_asr_curr)

    print(f"[Round {round_idx+1}] Global Acc: {global_acc_curr:.4f}, "
          f"Avg Train Loss: {avg_loss_this_round:.4f}, ASR: {global_asr_curr:.4f}")

# save results
save_results(test_acc_per_round, test_asr_per_round, train_loss_per_round, CONFIG["method"])
