import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision.datasets import ImageFolder

folder_mapping = lambda l: [1 if ll in [2,3,5,6,7,8,9,10,11,12,13,14,15,16] else 0 for ll in l]

class VIPDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = '../DeepfakeIEEE/original', batch_size: int = 64):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self):
        dataset = ImageFolder(self.data_dir)
        num_classes = max(dataset.targets) + 1

        # Determine which samples are real / fake
        is_fake_indices = folder_mapping(dataset.targets)
        real_labels = dataset.targets[is_fake_indices == 0]
        fake_labels = dataset.targets[is_fake_indices == 1]

        # Get total amount of real / fake samples
        real_class_samples = torch.unique(real_labels, return_counts=True)[1]
        fake_class_samples = torch.unique(fake_labels, return_counts=True)[1]

        real_class_weights = real_class_samples / num_classes
        fake_class_weights = fake_class_samples / num_classes

        # Calculate sampling weight for each folder
        class_weights = torch.zeros(num_classes)
        class_weights[folder_mapping(torch.arange(num_classes)) == 0] = real_class_weights
        class_weights[folder_mapping(torch.arange(num_classes)) == 1] = fake_class_weights
        print(class_weights)

        # Calculate amount of samples for each split
        train_samples = int(len(dataset) * 0.7)
        val_samples = int(len(dataset) * 0.1)
        test_samples = len(dataset) - train_samples - val_samples
        split_samples = (train_samples, val_samples, test_samples)

        # Create dataset for each split
        (train_dataset, train_indices), (val_dataset, val_indices), (test_dataset, test_indices)  = random_split(dataset, split_samples)

        self.train_dataset = train_dataset
        self.train_sampler = WeightedRandomSampler(class_weights[dataset.targets[train_indices]], train_samples)
        self.val_dataset = val_dataset
        self.val_sampler = WeightedRandomSampler(class_weights[dataset.targets[val_indices]], val_samples)
        self.test_dataset = test_dataset
        self.test_sampler = WeightedRandomSampler(class_weights[dataset.targets[test_indices]], test_samples)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=self.train_sampler)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, sampler=self.val_sampler)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, sampler=self.test_sampler)
