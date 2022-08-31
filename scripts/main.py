from pytorch_lightning.cli import LightningCLI
from src.data import VIPDataModule
from src.model import VIPModel

if __name__ == '__main__':
    cli = LightningCLI(VIPModel, VIPDataModule, save_config_overwrite=True)