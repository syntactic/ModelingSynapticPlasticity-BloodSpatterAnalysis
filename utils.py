import torch as th
import logging

logger = logging.getLogger('MSP_Project')

def get_device():
    """
    Returns the device to be used for computation.
    Returns:
        str: The device to be used for computation. Possible values are "cpu", "cuda:0", or "mps".
    """

    device = "cpu"
    if th.cuda.is_available():
        device = "cuda:0"
    elif th.backends.mps.is_available():
        device = "mps"
    return device

def smooth_k_fold_loss_record(loss_record):
    """
    Smooths the k-fold loss record by calculating the average loss for each epoch across all folds.
    Parameters:
    loss_record (list): A 2D list containing the loss values for each fold and epoch.
    Returns:
    list: A list of smoothed loss values for each epoch.
    """

    num_folds = len(loss_record)
    num_epochs = len(loss_record[0])
    smooth_k_fold_loss_record = []
    for epoch in range(num_epochs):
        epoch_losses = [loss_record[fold][epoch] for fold in range(num_folds)]
        smooth_k_fold_loss_record.append(sum(epoch_losses) / num_folds)
    return smooth_k_fold_loss_record

def is_spiking(model):
    """
    Check if the given model is a spiking model.
    Parameters:
        model: The model to be checked.
    Returns:
        bool: True if the model is a spiking model, False otherwise.
    """

    return 'Spiking' in model.__class__.__name__

device = get_device()