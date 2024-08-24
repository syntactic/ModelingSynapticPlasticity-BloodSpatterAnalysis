from torchvision.transforms import v2 as transforms
import logging 

logger = logging.getLogger('MSP_Project')
IMG_SIZE = 64 # fully connected size would be 64*13*13, but for 128 IMG_SIZE it should be 64*29*29
# List of data augmentation techniques
augmentations = [
    lambda x: x,
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(p=1),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
]

resize_transform = transforms.Resize((IMG_SIZE, IMG_SIZE))

preprocessing_no_normalize = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor() # Convert PIL Image to Tensor
])

def generate_preprocessing_transforms(mean, std):
    logger.debug("Using mean and std: " + str(mean) + ", " + str(std))
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(), # Convert PIL Image to Tensor
        transforms.Normalize(mean=mean, std=std)
    ])


# Preprocessing without tensor and normalization
preprocessing_no_tensor_no_normalize = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=1),
])

mnist_transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])