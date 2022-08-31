import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from dev.data import VIPDataModule
from dev.model import VIPModel

if __name__ == '__main__':
    model = VIPModel()
    data_module = VIPDataModule()

    trainer = pl.Trainer(epochs=1)
    trainer.fit(model, data_module)