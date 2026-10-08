import torch

CONFIG = {
    "DATA_FOLDER": "../FuGuard/data/IMAGE",
    "SAVE_DATA_FOLDER": "../FuGuard/backdoor/processed_IMAGE",
    "SAVE_RESULTS_FOLDER": "../FuGuard/results_image",
    "seed": 0,    # 0, 1, 42

    "dataset_name": 'SVHN',   # SVHN, EuroSAT, CIFAR10, CIFAR100
    "NUM_CLASSES": 10,
    "IMAGE_SIZE": (32, 32),
    "num_clients": 10,
    "dirichlet_alpha": 0.1,
    "inject_ratio": 1,

    "BACKDOOR_CLIENT": "client_3",
    "target_label": 9,

    "ROUNDS": 60,
    "EPOCHS_LOCAL": 5,
    "BATCH_SIZE": 256,
    "LR": 1e-3,

    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),

    # fuguard
    'samples_scale': 0.1,
    'gen_bs': 64,
    'unlearn_bs': 32,
    'ot_lambda': 0.1,

    'forgetting_epoch': 5,
    'forgetting_lr': 1e-3,

    "EXTERNAL_ALERTS": {
    "client_3": [50],
    },

    "method": 'fuguard',   # retrain, fedsga, fu, not, fast-fedul, fuguard

    # fu
    'distance_threshold': 2.2,
}


