import os, logging
from Transformations import *
from BloodDataset import BloodDataset
from Data import get_datasets
import numpy as np
import torch as th
from Models import *
from utils import *
from TrainingTesting import hyperparameter_search

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
    USING_MNIST = False

    train_set, test_set = get_datasets(in_colab, USING_MNIST)

    best_hyperparameters, best_accuracy = hyperparameter_search(train_set, using_mnist=USING_MNIST, spiking_model=True, num_epochs=5, batch_size=32, k=10)