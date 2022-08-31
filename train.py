import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from data import VIPDataModule
from model import VIPModel

if __name__ == '__main__':
    model = VIPModel()
    data_module = VIPDataModule(num_train_samples=20000, num_val_samples=1000)

    wandblogger = WandbLogger(project='vip-challenge')

    trainer = pl.Trainer(max_epochs=5, accelerator='gpu', devices=[1], logger=wandblogger)
    trainer.fit(model, data_module)