import torch as th
import logging

logger = logging.getLogger('MSP_Project')

def get_device():
    device = "cpu"
    if th.cuda.is_available():
        device = "cuda:0"
    elif th.backends.mps.is_available():
        device = "mps"
    return device

def smooth_k_fold_loss_record(loss_record):
    num_folds = len(loss_record)
    num_epochs = len(loss_record[0])
    smooth_k_fold_loss_record = []
    for epoch in range(num_epochs):
        epoch_losses = [loss_record[fold][epoch] for fold in range(num_folds)]
        smooth_k_fold_loss_record.append(sum(epoch_losses) / num_folds)
    return smooth_k_fold_loss_record

def is_spiking(model):
    return 'Spiking' in model.__class__.__name__

device = get_device()