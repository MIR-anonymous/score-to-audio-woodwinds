from torch import nn
import torch.optim as optim
from lightning import LightningModule
import torch.nn as nn
import torch.optim as optim

class PitchLigthning(LightningModule):
    def __init__(self, in_model):
        super().__init__()
        self.model = in_model
        self.save_hyperparameters(ignore=['in_model'])
        self.loss = nn.BCEWithLogitsLoss()

    def training_step(self, batch, batch_idx):
        hat_y_prob = self.model(batch['X'])
        loss = self.loss(hat_y_prob.squeeze(1), batch['y'])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        hat_y_prob = self.model(batch['X'])
        loss = self.loss(hat_y_prob.squeeze(1), batch['y'])
        self.log("valid_loss", loss)

    def test_step(self, batch, batch_idx):
        hat_y_prob = self.model(batch['X'])
        loss = self.loss(hat_y_prob.squeeze(1), batch['y'])
        self.log("test_loss", loss)
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), 0.0005)
        return optimizer

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)