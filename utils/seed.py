import random
import os
import numpy as np
import torch

def set_all_seeds(seed: int = 42):
    # Python 内置 random
    random.seed(seed)
    
    # Python 环境变量（影响某些库的行为）
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多 GPU
    
    # 为了进一步保证可复现性（但会降低性能）
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)