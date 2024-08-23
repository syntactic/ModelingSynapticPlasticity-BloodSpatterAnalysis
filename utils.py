import torch as th

def get_device():
    device = "cpu"
    if th.cuda.is_available():
        device = "cuda:0"
    elif th.backends.mps.is_available():
        device = "mps"
    return device