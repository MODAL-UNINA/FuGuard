#%%
import numpy as np
import random
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from geomloss import SamplesLoss
from FuGuard.main_awe.config import CONFIG
from FuGuard.main_awe.utils import (
    load_client_data, load_global_testdata,
    fedavg, evaluate_global, vae_loss, save_results, LSTMClassifier, LSTMVAE,
)

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


def request_unlearn(client_name, active_client_names, forget_set, global_model, clients):

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

    print("Loading VAE model...")
    seq_len = some_samples.shape[1]
    input_dim = some_samples.shape[2]
    gen_model = LSTMVAE(input_dim=input_dim, hidden_dim=256, latent_dim=32).to(device)
    print("VAE model loaded successfully.")
    optimizer = torch.optim.Adam(gen_model.parameters(), lr=1e-3)

    # train loop
    for epoch in range(1000):
        gen_model.train()
        total_loss = 0
        for signals, _ in sampled_dataloader:
            signals = signals.to(device)
            x_recon, mu, logvar = gen_model(signals)
            loss = vae_loss(signals, x_recon, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Gen Loss {total_loss/len(sampled_dataloader):.4f}")

    # Generate
    gen_model.eval()
    with torch.no_grad():
        mu, logvar = gen_model.encode(some_samples[:100].to(device))
        z = mu

        # PCA in latent space
        z_np = z.cpu().numpy()
        pca = PCA(n_components=5)
        pca.fit(z_np)
        principal_dirs = torch.tensor(pca.components_, dtype=torch.float32, device=device)

        z_new = apply_principal_direction(z, principal_dirs, dir_idx=1, alpha=1.0)

        new_data = gen_model.decode(z_new, seq_len=seq_len).clamp(0, 1)

    print(f"Generated new data with shape: {new_data.shape}")


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
