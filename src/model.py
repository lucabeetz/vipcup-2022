import timm
import torch
import torchmetrics
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class VIPModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # Create timm model
        self.model = timm.create_model('resnet18', pretrained=True, num_classes=2)

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

    def forward(self, x):
        x = self.model(x)
        return F.softmax(x, dim=-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        preds = y_hat.argmax(dim=-1)

        self.train_acc(preds, y)
        self.log('train_acc', self.train_acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        preds = y_hat.argmax(dim=-1)

        self.val_acc(preds, y)
        self.log('val_acc', self.val_acc, prog_bar=True)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-5)