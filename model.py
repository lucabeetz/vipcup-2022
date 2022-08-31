import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from data import folder_mapping

class VIPModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children)[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        self.classifier = nn.Linear(num_filters, 2)

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x)
        x = self.classifier(representations)
        return F.softmax(x, dim=-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        targets = F.one_hot(folder_mapping(y), 2)

        y_hat = self(x)
        loss = F.cross_entropy(y_hat, targets)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        targets = F.one_hot(folder_mapping(y), 2)

        y_hat = self(x)
        loss = F.cross_entropy(y_hat, targets)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Aadam(self.parameters(), lr=3e-4)