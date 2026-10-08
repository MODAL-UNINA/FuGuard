# utils.py
import os
import numpy as np
import torch
import torch.nn as nn
import pickle
from torch.utils.data import TensorDataset
from FuGuard.main_awe.config import CONFIG
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def load_client_data(client_name):
    path = os.path.join(CONFIG["SAVE_FOLDER"], f"{client_name}_train.npz")
    data = np.load(path, allow_pickle=True)
    train_X = data["train_X"].astype(np.float32)
    train_y = data["train_y"].astype(np.int64)

    train_dataset = TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y))
    return train_dataset


def load_global_testdata(backdoor=False):
    path = os.path.join(CONFIG["SAVE_FOLDER"], "global_test.npz")
    data = np.load(path, allow_pickle=True)
    
    if backdoor:
        X = data["X_backdoor"].astype(np.float32)
        y = data["y_backdoor"].astype(np.int64)
    else:
        X = data["X_clean"].astype(np.float32)
        y = data["y_clean"].astype(np.int64)
    
    # X = np.transpose(X, (0, 2, 1))  # (samples, channels, time)
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))


def evaluate_global(model, loader, device):
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
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }


# ------------------ LSTMClassifier ------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_classes=6, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [batch, seq_len, input_dim]
        output: [batch, num_classes]
        """
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)

    def embed(self, x):
        out, _ = self.lstm(x)           # [batch, seq_len, hidden_dim*2]
        out = out[:, -1, :]
        out = self.dropout(out)
        return out                      # [batch, hidden_dim*2]

class LSTMVAE(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, latent_dim=32, num_layers=1):
        super(LSTMVAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # ----- Encoder -----
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # ----- Decoder -----
        self.fc_dec = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        _, (h_n, _) = self.encoder_lstm(x)   # h_n: [num_layers, batch, hidden_dim]
        h_n = h_n[-1]                         # [batch, hidden_dim]
        mu = self.fc_mu(h_n)                  # [batch, latent_dim]
        logvar = self.fc_logvar(h_n)          # [batch, latent_dim]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, seq_len):
        h0 = torch.tanh(self.fc_dec(z)).unsqueeze(0)  # [1, batch, hidden_dim]
        c0 = torch.zeros_like(h0)                     # [1, batch, hidden_dim]

        # dummy input: zeros for each time step
        dec_input = torch.zeros(z.size(0), seq_len, self.hidden_dim, device=z.device)
        dec_out, _ = self.decoder_lstm(dec_input, (h0, c0))  # [batch, seq_len, hidden_dim]
        return self.output_layer(dec_out)  # [batch, seq_len, input_dim]

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, x.size(1))
        return x_recon, mu, logvar


def vae_loss(x, x_recon, mu, logvar):
    recon_loss = torch.nn.functional.mse_loss(x_recon, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + 0.1 * kl_loss


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
def save_results(test_acc_per_round, test_asr_per_round, train_loss_per_round, method):
    os.makedirs(CONFIG["SAVE_FOLDER"], exist_ok=True)

    file_path = os.path.join(CONFIG["SAVE_FOLDER"], f"{method}.pkl")

    with open(file_path, "wb") as f:
        pickle.dump({
            "test_acc_per_round": test_acc_per_round,
            "test_asr_per_round": test_asr_per_round,
            "train_loss_per_round": train_loss_per_round
        }, f)

    print("results saved!")
