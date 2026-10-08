import torch

CONFIG = {
    "DATA_FOLDER": "../FuGuard/data/AWE",
    "SAVE_DATA_FOLDER": "../FuGuard/backdoor/processed_AWE",
    "SAVE_RESULTS_FOLDER": "../FuGuard/results_awe",
    "F": 42,

    "FS_ORIG": 4000,
    "FS_TARGET": 500,
    "WINDOW_SIZE": 30,
    "STRIDE": 10,
    "LOWPASS_CUTOFF": 250,  # FS_TARGET / 2
    "FILTER_ORDER": 4,

    "num_clients": 10,
    "BACKDOOR_CLIENT": "client_2",
    "BACKDOOR_TARGET_CLASS": 5,
    "BACKDOOR_CHANNEL_IDX": 0,
    'BACKDOOR_TIME_IDX': 0,
    "BACKDOOR_TIME_RANGE": (0, 10),
    "BACKDOOR_AMPLITUDE": 5.0,

    "ROUNDS": 200,
    "EPOCHS_LOCAL": 2,
    "BATCH_SIZE": 32,
    "LR": 5e-5,

    "NUM_CLASSES": 6,
    "INPUT_CHANNELS": 1000,
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),

    'samples_scale': 0.1,
    'gen_bs': 64,
    'unlearn_bs': 64,
    'ot_lambda': 0.1,
    'alpha': 0.5,

    'forgetting_epoch': 5,
    'forgetting_lr': 0.05,

    "EXTERNAL_ALERTS": {
    "client_2": [100],
    },

    "method": 'fedsga',   # fuguard, retrain, 
    'distance_threshold': 2.2,
}


