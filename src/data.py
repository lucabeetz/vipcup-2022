import timm
import torch
import pytorch_lightning as pl
import pandas as pd
from typing import Optional
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from src.transform import ProvidedTransform
from src.utils import label_mapping, calculate_weights, DEFAULT_GROUPS
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

class VIPDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = '/mnt/hdd-data/DeepfakeIEEE/original', batch_size: int = 256, num_workers: int = 12, 
                num_train_samples: int = None, num_val_samples: int = None, test_data_path: str = None, timm_model_name: str = None):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.test_data_path = test_data_path

        # Load model transform
        if timm_model_name:
            model = timm.create_model(timm_model_name, pretrained=True, num_classes=2)
            config = resolve_data_config({}, model=model)
            final_transform = create_transform(**config)
        else:
            final_transform = transforms.ToTensor()

        self.transform = transforms.Compose([
            ProvidedTransform(),
            final_transform
        ])

    def setup(self, stage: Optional[str] = None):
        # Load test dataset and create list of files in it
        self.val_dataset = torch.load(self.test_data_path)
        self.val_dataset[0].dataset.transform = self.transform
        relative_val_filenames = [self.val_dataset[0].dataset.samples[i][0] for i in self.val_dataset[0].indices]
        val_filenames = set([f.replace("../","/mnt/hdd-data/DeepfakeIEEE/") for f in relative_val_filenames])

        # Create training dataset
        is_valid_file = lambda f: f not in val_filenames and f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
        self.train_dataset = ImageFolder(self.data_dir, self.transform, is_valid_file=is_valid_file)
        labels = torch.tensor(self.train_dataset.targets)

        weights = calculate_weights(labels)
        debug_df = pd.DataFrame({
            'labels': label_mapping(torch.arange(max(labels)+1)),
            'names': self.train_dataset.classes,
            'samples': torch.unique(labels, return_counts=True)[1],
            'group': DEFAULT_GROUPS,
            'weight': weights
        })

        self.train_sampler = WeightedRandomSampler(weights[labels], self.num_train_samples)
        self.val_sampler = WeightedRandomSampler(self.val_dataset[1], self.num_val_samples, generator=torch.Generator().manual_seed(0))

    def train_dataloader(self):
        return DataLoader(VIPDataset(self.train_dataset), batch_size=self.batch_size, sampler=self.train_sampler, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(VIPDataset(self.val_dataset[0]), batch_size=self.batch_size, sampler=self.val_sampler, num_workers=self.num_workers)

class VIPDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        target = label_mapping([label])[0]
        return img, target