import os, logging
from transformations import *
from BloodDataset import BloodDataset
from data import get_datasets
import numpy as np
import torch as th
from models import *
from utils import *
from training_testing import hyperparameter_search

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
    SPIKING_MODEL = False

    train_set, test_set = get_datasets(in_colab, USING_MNIST)

    best_hyperparameters, best_accuracy, best_models_loss_record, best_model = hyperparameter_search(train_set, using_mnist=USING_MNIST, spiking_model=False, num_epochs=10, batch_size=32, k=10)