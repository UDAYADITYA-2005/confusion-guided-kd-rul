import torch
import random
import numpy as np

# Global Hyperparameters
MAX_RUL = 130
WIN_SIZE = 30
N_FEATURES = 14

# Device Configuration
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=99):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
