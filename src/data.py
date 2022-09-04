from cgi import test
from typing import Optional
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from src.transform import ProvidedTransform
from src.utils import label_mapping, calculate_weights

class VIPDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = '/mnt/hdd-data/DeepfakeIEEE/original', batch_size: int = 256, num_workers: int = 12,
                num_train_samples: int = None, num_val_samples: int = None, test_data_path: str = None):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.test_data_path = test_data_path

        self.transform = transforms.Compose([
            ProvidedTransform(),
            transforms.CenterCrop(200),
            transforms.ToTensor()
        ])

    def setup(self, stage: Optional[str] = None):
        # Create list of filenames for test images
        # self.test_dataset = torch.load(self.test_data_path)
        # test_filenames = [self.test_dataset[0].dataset.samples[i][0] for i in self.test_dataset[0].indices]

        # Create dataset with train and val data
        # is_valid_file = lambda f: f not in test_filenames and f.lower().endswith(('.png', '.jpg'))
        dataset = ImageFolder(self.data_dir, self.transform)

        labels = torch.tensor(dataset.targets)
        weights = calculate_weights(labels)

        # Calculate amount of samples for each split
        train_samples = int(len(dataset) * 0.8)
        val_samples = len(dataset) - train_samples

        # Create train and val datasets
        self.train_dataset, self.val_dataset = random_split(dataset, (train_samples, val_samples))

        self.train_sampler = WeightedRandomSampler(class_weights[labels[self.train_dataset.indices]], self.num_train_samples, replacement=True)
        self.val_sampler = WeightedRandomSampler(class_weights[labels[self.val_dataset.indices]], self.num_val_samples, replacement=True)

    def train_dataloader(self):
        return DataLoader(VIPDataset(self.train_dataset), batch_size=self.batch_size, sampler=self.train_sampler, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(VIPDataset(self.val_dataset), batch_size=self.batch_size, sampler=self.val_sampler, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(VIPDataset(self.test_dataset), batch_size=self.batch_size, num_workers=self.num_workers)

class VIPDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        target = label_mapping([label])[0]
        return img, target