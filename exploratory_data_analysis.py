import matplotlib.pyplot as plt
import numpy as np
import torchvision
import logging
from torch.utils.data import DataLoader
from data import get_class_counts, get_classes

logger = logging.getLogger('MSP_Project')

# Function to show an image
def imshow(img, title):
    """
    Display an image using matplotlib.
    Parameters:
    img (torch.Tensor): The image tensor to be displayed.
    title (str): The title of the image.
    """

    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()

# Calculate the average image of each class
def calculate_class_average(dataset):
    """
    Calculate and plot the average image for each class in the dataset.
    Args:
        dataset (list): A list of tuples containing image and label pairs.
    """

    classes = get_classes(dataset)
    class_sums = [0] * len(classes)
    class_counts = [0] * len(classes)

    for img, label in dataset:
        class_sums[label] += img
        class_counts[label] += 1

    class_averages = [class_sums[i] / class_counts[i] for i in range(len(classes))]

    fig, axes = plt.subplots(1, len(classes), figsize=(20, 5))
    for i, avg_img in enumerate(class_averages):
        axes[i].imshow(avg_img.squeeze(), cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(classes[i])
    plt.show()

def run_eda(dataset):
    """
    Perform exploratory data analysis on the given dataset. This includes visualizing images 
    and plotting class distributions.
    Args:
        dataset: The dataset to be analyzed.
    """

    # Define DataLoader for EDA
    eda_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Visualize some images
    dataiter = iter(eda_loader)
    images, labels = next(dataiter)  # Use next(dataiter) instead of dataiter.next()

    # Show images
    imshow(torchvision.utils.make_grid(images[:4]), title=[dataset.classes[label] for label in labels[:4]])

    # Calculate class counts
    class_counts = get_class_counts(dataset)
    classes_sorted = get_classes(dataset)
    logger.debug("Dataset classes:" + str(classes_sorted))
    logger.debug("Class Counts: " + str(class_counts))
    
    plt.bar(classes_sorted, [class_counts[cls] for cls in classes_sorted])
    plt.title('Class Distribution in Dataset')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.show()