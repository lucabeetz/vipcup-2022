import timm
import torch
import torchmetrics
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class VIPModel(pl.LightningModule):
    def __init__(self, timm_model_name: str, learning_rate: float, weight_decay: float):
        super().__init__()
        
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Create timm model
        self.model = timm.create_model(timm_model_name, pretrained=True, num_classes=2)
        nn.init.zeros_(self.model.head.weight) #added

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        preds = y_hat.argmax(dim=-1)

        self.train_acc(preds, y)
        self.log('train/acc', self.train_acc, prog_bar=True)
        self.log('train/loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        preds = y_hat.argmax(dim=-1)

        self.val_acc(preds, y)
        self.log('val/acc', self.val_acc, prog_bar=True)
        self.log('val/loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        preds = y_hat.argmax(dim=-1)

        self.test_acc.update(preds, y)
        self.log('test/acc', self.test_acc)
        self.log('test/loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5),
            'monitor': 'val/loss'
        }
        return [optimizer], [lr_scheduler]