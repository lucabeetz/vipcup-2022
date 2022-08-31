import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from data import VIPDataModule
from model import VIPModel

if __name__ == '__main__':
    model = VIPModel()
    data_module = VIPDataModule()

    trainer = pl.Trainer(max_epochs=1, accelerator='gpu', devices=[1])
    trainer.fit(model, data_module)