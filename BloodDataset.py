from torchvision.datasets import ImageFolder

class BloodDataset(ImageFolder):
    def __init__(self, root, init_transform=None, transform=lambda x : x, augmentations=[]):
        super().__init__(root, transform=transform)
        if init_transform is not None:
            for i, (img_path, target) in enumerate(self.samples):
                data = self.loader(img_path)
                self.samples[i] = (init_transform(data), target)
        self.augmentations = augmentations
        self.augmentation_applied = False

    def __len__(self):
        return len(self.samples)

    def from_subset(self, subset):
        subset_samples = subset.dataset.samples
        subset_indices = subset.indices
        self.samples = [subset.dataset.samples[i] for i in subset_indices]
        self.targets = [s[1] for s in self.samples]
        self.augmentations = subset.dataset.augmentations
        self.augmentation_applied = subset.dataset.augmentation_applied
        self.transform = subset.dataset.transform
        self.target_transform = subset.dataset.target_transform

    # this function takes the original samples which are (path, label) pairs
    # and turns them into ((path, augmentation), label) pairs which would get
    # executed at data retrieval time
    def apply_augmentations(self):
        if self.augmentation_applied:
            print("Already augmented the dataset. Will skip.")
            return
        augmented_samples = []
        for sample, label in self.samples:
            for augmentation in self.augmentations:
                augmented_samples.append(((sample, augmentation), label))
        self.samples = augmented_samples
        self.targets = [s[1] for s in self.samples]
        self.augmentation_applied = True

    def __getitem__(self, index):
        data, target = self.samples[index]
        if type(data) == tuple:
            img_data, augmentation = data
            sample = augmentation(img_data)
        else:
            sample = data

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
