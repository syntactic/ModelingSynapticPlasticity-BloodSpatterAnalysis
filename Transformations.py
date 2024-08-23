from torchvision.transforms import v2 as transforms

# List of data augmentation techniques
augmentations = [
    lambda x: x,
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
]

resize_transform = transforms.Resize((128, 128))

preprocessing = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(), # Convert PIL Image to Tensor
    transforms.Normalize(mean=[0.5], std=[0.5])
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