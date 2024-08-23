import os, logging
from Transformations import *
from BloodDataset import BloodDataset
from Data import get_datasets
import numpy as np
import torch as th
import itertools
from Models import *
from TrainingTesting import k_fold_cross_validation
from utils import *

device = get_device()

def hyperparameter_search(train_set, using_mnist=False, spiking_model=True, num_epochs=5, batch_size=16, k=10):
    learning_rates = [1e-3, 1e-4, 1e-5]
    weight_decays = [1e-3, 1e-4, 1e-5]

    hyperparameter_types = [learning_rates, weight_decays]
    if spiking_model:
        betas = [0.8, 0.9, 0.95]
        hyperparameter_types.append(betas)
    
    hyperparameter_combinations = list(itertools.product(*hyperparameter_types))
    logger.debug(f"Hyperparameter combinations: {hyperparameter_combinations}")

    best_accuracy = 0
    best_hyperparameters = None
    
    for i, hyperparameter in enumerate(hyperparameter_combinations):
        logger.debug(f"Hyperparameter combination {i+1}/{len(hyperparameter_combinations)}: {hyperparameter}")
        learning_rate = hyperparameter[0]
        weight_decay = hyperparameter[1]
        if spiking_model:
            beta = hyperparameter[2]
            model = SpikingCNN(beta=beta)
        else:
            model = PyTorchCNN(using_mnist=using_mnist)
        model = model.to(device)
        optimizer = th.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        mean_accuracy = k_fold_cross_validation(train_set, model, k=k, num_epochs=num_epochs, optimizer=optimizer, batch_size=batch_size, verbose=True)
        logger.debug(f"Mean accuracy: {mean_accuracy}")
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_hyperparameters = hyperparameter

    logger.debug(f"Best hyperparameters: {best_hyperparameters}")
    logger.debug(f"Best accuracy: {best_accuracy}")
    return best_hyperparameters, best_accuracy

if __name__ == '__main__':
    np.random.seed(0)
    th.manual_seed(0)
    logger = logging.getLogger('MSP_Project')
    logging.basicConfig(level=logging.DEBUG)
    logger.setLevel(logging.DEBUG)

    in_colab = False
    if os.getenv("COLAB_RELEASE_TAG"):
        in_colab = True
        logger.debug("Running in Colab")
    USING_MNIST = True

    train_set, test_set = get_datasets(in_colab, USING_MNIST)

    best_hyperparameters, best_accuracy = hyperparameter_search(train_set, using_mnist=USING_MNIST, spiking_model=False, num_epochs=5, batch_size=64, k=10)