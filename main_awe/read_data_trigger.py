#%% ------------------ Imports & CONFIG ------------------
import numpy as np
import json
import os
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from FuGuard.main_awe.config import CONFIG


os.makedirs(CONFIG["SAVE_FOLDER"], exist_ok=True)
rng = np.random.default_rng(CONFIG["SEED"])

STATE_GROUPS = {0:7,1:8,2:8,3:8,4:8,5:8}
TRAIN_GROUPS = {
    0:[0,1,2,3,4,5],
    1:[0,1,2,3,4,5,6],
    2:[0,1,2,3,4,5,6],
    3:[0,1,2,3,4,5,6],
    4:[0,1,2,3,4,5,6],
    5:[0,1,2,3,4,5,6]
}
TEST_GROUPS = {0:[6],1:[7],2:[7],3:[7],4:[7],5:[7]}
PRIMARY_GROUPS = {0:[0,1,2,3],1:[0,1,2,3],2:[0,1,2,3],3:[0,1,2,3],4:[0,1,2,3],5:[0,1,2,3]}
STATE_TO_CLIENT = {0:"client_0", 1:"client_1", 2:"client_2", 3:"client_3", 4:"client_4", 5:"client_5"}

def load_dict(filename):
    with open(filename, "r") as f:
        return json.load(f)

def load_data(filename):
    data_dict = load_dict(filename)
    vibr_UB, vibr_LB = [], []
    for j in range(len(data_dict)):
        vibr_UB.append(data_dict[str(j)]['vibr_UB'])
        vibr_LB.append(data_dict[str(j)]['vibr_LB'])
    return np.c_[vibr_UB, vibr_LB]

def zscore_normalize(data):
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    return (data - mean) / std

def lowpass_iir(data, cutoff=CONFIG["LOWPASS_CUTOFF"], fs=CONFIG["FS_ORIG"], order=CONFIG["FILTER_ORDER"]):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=1)

def downsample(data, factor=CONFIG["FS_ORIG"]//CONFIG["FS_TARGET"]):
    return data[:, ::factor]

def create_windows(data, window_size=CONFIG["WINDOW_SIZE"], stride=CONFIG["STRIDE"]):
    windows = []
    for start in range(0, data.shape[0] - window_size + 1, stride):
        windows.append(data[start:start+window_size, :])
    return np.array(windows)

def generate_labels(num_windows, state):
    return np.full(num_windows, state)


# backdoor
def add_spike_trigger(
    X_np,
    channel_idx=CONFIG["BACKDOOR_CHANNEL_IDX"],
    time_idx=CONFIG["BACKDOOR_TIME_IDX"],
    amplitude=CONFIG["BACKDOOR_AMPLITUDE"],
    extra_channels=None,
    extra_times=None,
    noise_std=0.0
):
    """
    Returns:
        X_poisoned: numpy array, same shape as X_np
    """
    X = X_np.copy()
    N, C, T = X.shape

    X[:, channel_idx, time_idx] += amplitude

    if extra_channels is not None:
        for ch in extra_channels:
            X[:, ch, time_idx] += amplitude

    if extra_times is not None:
        for t in extra_times:
            X[:, channel_idx, t] += amplitude
            if extra_channels is not None:
                for ch in extra_channels:
                    X[:, ch, t] += amplitude

    if noise_std > 0.0:
        X += np.random.normal(0, noise_std, X.shape)

    return X


client_data = {c: {"train": [], "train_labels": []} for c in CONFIG["CLIENTS"]}
global_test_set, global_test_labels = [], []

client_data_clean = {c: {"train": [], "train_labels": []} for c in CONFIG["CLIENTS"]}

for state in STATE_GROUPS.keys():
    for group in TRAIN_GROUPS[state]:
        filename = os.path.join(CONFIG["DATA_FOLDER"], f"{state}chips", f"{state}chips_{group}.json")
        if not os.path.exists(filename):
            continue

        data = load_data(filename)
        data = zscore_normalize(data)
        data = lowpass_iir(data)
        data = downsample(data)

        train_windows = create_windows(data)
        train_labels = generate_labels(len(train_windows), state)

        if group in PRIMARY_GROUPS[state]:
            client = STATE_TO_CLIENT[state]
        else:
            client = rng.choice(CONFIG["CLIENTS"])

        client_data_clean[client]["train"].append(train_windows)
        client_data_clean[client]["train_labels"].append(train_labels)

        if client == CONFIG["BACKDOOR_CLIENT"]:

            train_windows = add_spike_trigger(
                train_windows,
                amplitude=20,
                extra_channels=[1,2],
                extra_times=[5,6,7],
                noise_std=0.5
            )

            train_labels = np.full_like(train_labels, CONFIG["BACKDOOR_TARGET_CLASS"])

        client_data[client]["train"].append(train_windows)
        client_data[client]["train_labels"].append(train_labels)

    for group in TEST_GROUPS[state]:
        filename = os.path.join(CONFIG["DATA_FOLDER"], f"{state}chips", f"{state}chips_{group}.json")
        if not os.path.exists(filename):
            continue

        data = load_data(filename)
        data = zscore_normalize(data)
        data = lowpass_iir(data)
        data = downsample(data)
        windows = create_windows(data)
        labels = generate_labels(len(windows), state)

        global_test_set.append(windows)
        global_test_labels.append(labels)


for client in CONFIG["CLIENTS"]:
    client_data[client]["train_X"] = np.concatenate(client_data[client]["train"], axis=0)
    client_data[client]["train_y"] = np.concatenate(client_data[client]["train_labels"], axis=0)
    print(f"{client} 总训练样本数={client_data[client]['train_X'].shape[0]}")


global_test_X = np.concatenate(global_test_set, axis=0)
global_test_y = np.concatenate(global_test_labels, axis=0)

mask = (global_test_y != CONFIG["BACKDOOR_TARGET_CLASS"])
# X_triggered = add_spike_trigger(global_test_X[mask])
X_triggered = add_spike_trigger(
    global_test_X[mask],
                amplitude=20,
                extra_channels=[1,2],
                extra_times=[5,6,7],
                noise_std=0.5 
)
y_triggered = np.full(len(X_triggered), CONFIG["BACKDOOR_TARGET_CLASS"], dtype=global_test_y.dtype)


for client in CONFIG["CLIENTS"]:
    np.savez(os.path.join(CONFIG["SAVE_FOLDER"], f"{client}_train.npz"),
             train_X=client_data[client]["train_X"],
             train_y=client_data[client]["train_y"])

np.savez(os.path.join(CONFIG["SAVE_FOLDER"], "global_test.npz"),
         X_clean=global_test_X,
         y_clean=global_test_y,
         X_backdoor=X_triggered,
         y_backdoor=y_triggered)

print("saved！")

def plot_client_data_distribution(client_data, clients, subset="train"):
    """
    subset: "train"
    """
    key_y = f"{subset}_y"
    all_labels = sorted({label for client in clients for label in np.unique(client_data[client][key_y])})
    bar_width = 0.6
    colors = plt.cm.tab10.colors
    client_indices = np.arange(len(clients))
    bottoms = np.zeros(len(clients))

    plt.figure(figsize=(10,6))
    for i, label in enumerate(all_labels):
        counts = []
        for client in clients:
            y = client_data[client][key_y]
            counts.append(np.sum(y == label))
        counts = np.array(counts)
        plt.bar(client_indices, counts, bottom=bottoms, width=bar_width,
                color=colors[i % 10], label=f"class {label}")
        bottoms += counts

    plt.xticks(client_indices, clients)
    plt.ylabel(f"Number of {subset} samples")
    plt.title(f"Client {subset.capitalize()} Sample Distribution by Class")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

plot_client_data_distribution(client_data, CONFIG["CLIENTS"], subset="train")


def plot_global_test_distribution(global_test_X, global_test_y, X_backdoor, y_backdoor):
    unique_clean, counts_clean = np.unique(global_test_y, return_counts=True)
    unique_bd, counts_bd = np.unique(y_backdoor, return_counts=True)

    plt.figure(figsize=(10,5))
    plt.bar(unique_clean - 0.2, counts_clean, width=0.4, label="Clean Test", color='lightgreen')
    plt.bar(unique_bd + 0.2, counts_bd, width=0.4, label="Backdoor Test", color='salmon')
    plt.xlabel("Class Label")
    plt.ylabel("Number of samples")
    plt.title("Global Test Set Distribution (Clean vs Backdoor)")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

plot_global_test_distribution(global_test_X, global_test_y, X_triggered, y_triggered)

# %%
def plot_backdoor_signal(client_name, window_idx=0, channel_idx=0, time_length=30,
                         client_data_clean=None, client_data=None, train_idx=0):

    if client_data_clean is None or client_data is None:
        raise ValueError("请提供 client_data_clean 和 client_data")

    clean_array = client_data_clean[client_name]["train"][train_idx]
    clean_window = clean_array[window_idx, :time_length, channel_idx]

    backdoor_array = client_data[client_name]["train"][train_idx]
    backdoor_window = backdoor_array[window_idx, :time_length, channel_idx]

    plt.figure(figsize=(10,4))
    plt.plot(clean_window, label="Original Signal", color='blue')
    plt.plot(backdoor_window, label="Backdoor Signal", color='red', linestyle='--')
    plt.xlabel("Time Step")
    plt.ylabel("Amplitude")
    plt.title(f"{client_name} Window {window_idx} - Signal Before/After Backdoor")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_backdoor_signal(
    client_name='client_2',
    window_idx=0,
    channel_idx=0,
    time_length=30,
    client_data_clean=client_data_clean,
    client_data=client_data,
    train_idx=0
)


def plot_backdoor_signal_heatmap(client_name, window_idx=0, time_length=None,
                                 client_data_clean=None, client_data=None,
                                 train_idx=0, num_channels=3, cmap_clean='plasma_r', cmap_backdoor='plasma_r'):

    if client_data_clean is None or client_data is None:
        raise ValueError("please provide client_data_clean and client_data")
    
    clean_array = client_data_clean[client_name]["train"][train_idx][window_idx]
    backdoor_array = client_data[client_name]["train"][train_idx][window_idx]
    
    if time_length is None:
        time_length = clean_array.shape[0]

    Z_clean = np.array([clean_array[:time_length, ch] for ch in range(num_channels)])
    Z_backdoor = np.array([backdoor_array[:time_length, ch] for ch in range(num_channels)])
    
    vmin = min(Z_clean.min(), Z_backdoor.min())
    vmax = max(Z_clean.max(), Z_backdoor.max())

    fig, axs = plt.subplots(1, 2, figsize=(7, 3),dpi=500)

    im1 = axs[0].imshow(Z_clean, aspect='auto', origin='lower', cmap=cmap_clean, vmin=vmin, vmax=vmax)
    axs[0].set_title("Clean Signal", fontsize=12)
    axs[0].set_xlabel("Time Step", fontsize=12)
    axs[0].set_ylabel("Feature Index", fontsize=12)
    fig.colorbar(im1, ax=axs[0])

    im2 = axs[1].imshow(Z_backdoor, aspect='auto', origin='lower', cmap=cmap_backdoor, vmin=vmin, vmax=vmax)
    axs[1].set_title("Triggered Signal", fontsize=12)
    axs[1].set_xlabel("Time Step", fontsize=12)
    axs[1].set_ylabel("Feature Index", fontsize=12)
    fig.colorbar(im2, ax=axs[1])

    plt.tight_layout()
    plt.show()


plot_backdoor_signal_heatmap(
    client_name='client_2',
    window_idx=0,
    time_length=100,
    client_data_clean=client_data_clean,
    client_data=client_data,
    train_idx=0,
    num_channels=500
)

