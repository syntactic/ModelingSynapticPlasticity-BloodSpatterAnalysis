import matplotlib.pyplot as plt
import numpy as np
import torchvision
import logging
from torch.utils.data import DataLoader
from Data import get_class_counts, get_classes

logger = logging.getLogger('MSP_Project')

# Function to show an image
def imshow(img, title):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()

# Calculate the average image of each class
def calculate_class_average(dataset):
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
    # get class counts sorted by class name numerically
    #classes_sorted = sorted(class_counts.keys(), key=lambda x: int(x))
    #class_counts = [class_counts[cls] for cls in classes_sorted]
    logger.debug("Class Counts: " + str(class_counts))
    
    plt.bar(classes_sorted, [class_counts[cls] for cls in classes_sorted])
    plt.title('Class Distribution in Dataset')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.show()