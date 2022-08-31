from typing import Optional
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from transform import ProvidedTransform

folder_mapping = lambda l: torch.tensor([1 if ll in [2,3,5,6,7,8,9,10,11,12,13,14,15,16] else 0 for ll in l])

class VIPDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = '/mnt/hdd-data/DeepfakeIEEE/original', batch_size: int = 256, num_workers: int = 12,
                num_train_samples: int = None, num_val_samples: int = None):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples

        self.transform = transforms.Compose([
            ProvidedTransform(),
            transforms.CenterCrop(200),
            transforms.ToTensor()
        ])

    def setup(self, stage: Optional[str] = None):
        dataset = ImageFolder(self.data_dir, self.transform)

        labels = torch.tensor(dataset.targets)
        num_classes = max(dataset.targets) + 1

        # Determine which samples are real / fake
        is_fake_indices = folder_mapping(dataset.targets)
        real_labels = labels[is_fake_indices == 0]
        fake_labels = labels[is_fake_indices == 1]

        # Get total amount of real / fake samples
        real_class_samples = torch.unique(real_labels, return_counts=True)[1]
        fake_class_samples = torch.unique(fake_labels, return_counts=True)[1]

        real_class_weights = 1 / real_class_samples / len(real_class_samples)
        fake_class_weights = 1 / fake_class_samples / len(fake_class_samples)

        # Calculate sampling weight for each folder
        class_weights = torch.zeros(num_classes)
        class_weights[folder_mapping(torch.arange(num_classes)) == 0] = real_class_weights
        class_weights[folder_mapping(torch.arange(num_classes)) == 1] = fake_class_weights
        print(class_weights)

        # Calculate amount of samples for each split
        train_samples = int(len(dataset) * 0.8)
        val_samples = len(dataset) - train_samples

        # Create train and val datasets
        self.train_dataset, self.val_dataset = random_split(dataset, (train_samples, val_samples))

        self.train_sampler = WeightedRandomSampler(class_weights[labels[self.train_dataset.indices]], self.num_train_samples)
        self.val_sampler = WeightedRandomSampler(class_weights[labels[self.val_dataset.indices]], self.num_val_samples)

        # TODO: Add test dataset

    def train_dataloader(self):
        return DataLoader(VIPDataset(self.train_dataset), batch_size=self.batch_size, sampler=self.train_sampler, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(VIPDataset(self.val_dataset), batch_size=self.batch_size, sampler=self.val_sampler, num_workers=self.num_workers)

    def test_dataloader(self):
        pass

class VIPDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        target = folder_mapping([label])[0]
        return img, target