from torchvision import datasets
import gdown, shutil, os, logging
from Transformations import *
from BloodDataset import BloodDataset
from torch.utils.data import random_split
import torch as th

BLOOD_DATASET_FILENAME = 'full_dataset.zip'
FILE_ID = "1XwfVHx4IU4srDZJLiCFLY8-3UtnwOWxQ"
logger = logging.getLogger('MSP_Project')

def download_data(in_colab=False, using_mnist=False, data_dir='./'):
    if using_mnist:
        data_dir = os.path.join(data_dir, 'tmp', 'data', 'mnist')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    else:
        gdown.download(id=FILE_ID)
        shutil.unpack_archive(BLOOD_DATASET_FILENAME, data_dir)

def calculate_mean_and_std(train_set):
    loader = th.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data, _ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples
    mean /= nb_samples
    std /= nb_samples
    return mean, std

def get_datasets(in_colab=False, using_mnist=False):
    if in_colab:
        data_dir = '/'
    else:
        data_dir = './'
    if using_mnist:
        data_dir = os.path.join(data_dir, 'tmp', 'data', 'mnist')
    else:
        data_dir = os.path.join(data_dir, 'full_dataset')
    logger.debug("Data directory: " + data_dir)
    if not os.path.exists(data_dir):
        download_data(in_colab, using_mnist, data_dir)

    if using_mnist:
        train_set = datasets.MNIST(data_dir, train=True, download=True, transform=mnist_transform)
        test_set = datasets.MNIST(data_dir, train=False, download=True, transform=mnist_transform)
    else:
        throwaway_blood_dataset = BloodDataset(data_dir, init_transform=resize_transform,
                                                transform=preprocessing_no_normalize)
        mean, std = calculate_mean_and_std(throwaway_blood_dataset)
        full_blood_dataset = BloodDataset(data_dir, init_transform=resize_transform, 
                                          transform=generate_preprocessing_transforms(mean, std), augmentations=augmentations)
        representative_test_set = False
        class_percentages = get_class_percentages(full_blood_dataset)
        while not representative_test_set:
            logger.debug("Trying to get a representative test set...")
            train_set_unaugmented, test_set = random_split(full_blood_dataset, [0.8, 0.2])
            if matches_proportions(test_set, class_percentages):
                representative_test_set = True
                logger.debug(str(len(train_set_unaugmented)) + " images in training set before augmentation.")
            train_set = BloodDataset(data_dir)
            train_set.from_subset(train_set_unaugmented)
            train_set.apply_augmentations()
    logger.debug(str(len(train_set)) + " images in training set.")
    logger.debug(str(len(test_set)) + " images in test set.")
    return train_set, test_set
        
def get_classes(dataset):
    class_counts = get_class_counts(dataset)
    return sorted(class_counts.keys())

def get_class_counts(dataset):
    class_counts = {}
    for _, label in dataset:
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1
    return class_counts

def get_class_percentages(dataset):
    class_counts = get_class_counts(dataset)
    class_percentages = {}
    for label, count in class_counts.items():
        class_percentages[label] = count / len(dataset)
    return class_percentages

def matches_proportions(test_set, class_percentages, tolerance=0.1):
    num_classes = len(class_percentages) # induce number of classes from class_percentages
    test_class_percentages = get_class_percentages(test_set)
    for label in range(num_classes):
        if abs(test_class_percentages[label] - class_percentages[label]) > tolerance:
            return False
    return True