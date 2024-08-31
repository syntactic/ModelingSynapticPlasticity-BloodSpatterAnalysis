from torchvision import datasets
import gdown, shutil, os, logging
from transformations import *
from BloodDataset import BloodDataset
from torch.utils.data import random_split
import torch as th

BLOOD_DATASET_FILENAME = 'full_dataset.zip'
FILE_ID = "1XwfVHx4IU4srDZJLiCFLY8-3UtnwOWxQ"
logger = logging.getLogger('MSP_Project')

def download_data(using_mnist=False, data_dir='./'):
    """
    Downloads and extracts the data for the project.
    Parameters:
        using_mnist (bool): Flag indicating whether to use the MNIST dataset. 
        data_dir (str): The directory where the data will be downloaded and extracted.
    """

    if using_mnist:
        data_dir = os.path.join(data_dir, 'tmp', 'data', 'mnist')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    else:
        gdown.download(id=FILE_ID)
        shutil.unpack_archive(BLOOD_DATASET_FILENAME, data_dir)

def calculate_mean_and_std(train_set):
    """
    Calculates the mean and standard deviation of the given training set.
    Parameters:
        train_set (torch.utils.data.Dataset): The training set.
    Returns:
        tuple: A tuple containing the mean and standard deviation of the training set.
    """
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

def get_datasets(using_mnist=False):
    """
    Retrieves the training and test datasets for the project.
    Parameters:
        using_mnist (bool, optional): Indicates whether to use the MNIST dataset. 
            Defaults to False.
    Returns:
        tuple: A tuple containing the training set and test set.
    """
    # Function code here

    data_dir = './'
    if using_mnist:
        data_dir = os.path.join(data_dir, 'tmp', 'data', 'mnist')
    else:
        data_dir = os.path.join(data_dir, 'full_dataset')
    logger.debug("Data directory: " + data_dir)
    if not os.path.exists(data_dir):
        download_data(using_mnist, data_dir)

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
        
def get_classes(dataset, from_original=False):
    """
    Returns a sorted list of classes present in the given dataset.
    Parameters:
        dataset (Dataset): The dataset object containing the data.
        from_original (bool, optional): If True, returns classes from the original dataset. 
                                         If False (default), returns classes based on class counts.
    Returns:
        list: A sorted list of classes present in the dataset.
    """

    if from_original:
        return sorted(dataset.dataset.class_to_idx.keys())
    class_counts = get_class_counts(dataset)
    return sorted(class_counts.keys())

def get_class_counts(dataset):
    """
    Calculates the count of each class in the given dataset.
    Parameters:
        dataset (torch.utils.data.Dataset): The dataset containing the samples and labels.
    Returns:
        dict: A dictionary where the keys are the class labels and the values are the counts of each class.
    """

    original_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset
    class_to_idx = original_dataset.class_to_idx.items()
    idx_to_class = {v: k for k, v in class_to_idx}
    class_counts = {}
    for _, label in dataset:
        label = idx_to_class[label]
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1
    return class_counts

def get_class_percentages(dataset):
    """
    Calculate the percentage of each class in the given dataset.
    Parameters:
        dataset (list): A list of data points representing the dataset.
    Returns:
        class_percentages (dict): A dictionary containing the percentage of each class in the dataset.
    """

    class_counts = get_class_counts(dataset)
    class_percentages = {}
    for label, count in class_counts.items():
        class_percentages[label] = count / len(dataset)
    logger.debug("Class percentages: " + str(class_percentages))
    return class_percentages

def matches_proportions(test_set, class_percentages, tolerance=0.1):
    """
    Check if the proportions of classes in the test set match the given class percentages within a tolerance.
    Parameters:
        test_set (torch.utils.data.Dataset): The test set containing the data to be checked.
        class_percentages (dict): A dictionary mapping class labels to their expected percentages in the test set.
        tolerance (float, optional): The maximum allowed difference between the expected and actual class percentages. Defaults to 0.1.
    Returns:
        bool: True if the proportions match within the tolerance, False otherwise.
    """

    classes = get_classes(test_set, from_original=True)
    test_class_percentages = get_class_percentages(test_set)
    for label in classes:
        if abs(test_class_percentages[label] - class_percentages[label]) > tolerance:
            return False
    return True