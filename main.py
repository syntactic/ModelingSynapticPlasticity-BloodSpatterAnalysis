import os, logging
from transformations import *
from data import get_datasets
import numpy as np
import torch as th
from models import *
from utils import *
from training_testing import hyperparameter_search, train_and_test

def main():
    np.random.seed(0)
    th.manual_seed(0)
    logger = logging.getLogger('MSP_Project')
    logging.basicConfig(level=logging.DEBUG)
    logger.setLevel(logging.DEBUG)

    USING_MNIST = False
    SPIKING_MODEL = False

    train_set, test_set = get_datasets(USING_MNIST)

    best_hyperparameters, best_accuracy, best_models_loss_record, best_model = hyperparameter_search(train_set, using_mnist=USING_MNIST, spiking_model=SPIKING_MODEL, num_epochs=10, batch_size=32, k=10)
    
    if SPIKING_MODEL:
        best_beta = best_hyperparameters[0]
    best_learning_rate, best_weight_decay = best_hyperparameters[-2:]

    logger.debug("Best hyperparameters: " + str(best_hyperparameters))
    logger.debug("Best accuracy: " + str(best_accuracy))

    if SPIKING_MODEL:
        model = SpikingCNNSerial(beta=best_beta, using_mnist=USING_MNIST)
    else:
        model = PyTorchCNN(using_mnist=USING_MNIST)
    model.to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=best_learning_rate, weight_decay=best_weight_decay)

    training_losses, training_accuracies, testing_losses, testing_accuracies = train_and_test(model, optimizer, train_set, 
                                                                                             test_set, num_epochs=25, batch_size=32, verbose=True)

if __name__ == '__main__':
    main()
