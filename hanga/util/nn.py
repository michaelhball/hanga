import gc

import torch


def clear_mem():
    torch.cuda.empty_cache()
    gc.collect()
