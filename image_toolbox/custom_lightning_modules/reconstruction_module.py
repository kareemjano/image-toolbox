import torch
from torch import nn
import pytorch_lightning as pl
from abc import ABC

class ReconstructionModule(pl.LightningModule, ABC):
    def __init__(self, cfg):
        """
        configuration parameters containing params and hparams.
        """
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg
        self.dataset_cfg = cfg.dataset
        self.model_select = cfg.nets.select
        self.net_params = cfg.nets[self.model_select]
        self.model_params = self.net_params.params
        self.hparam = self.net_params.hparams

        # CNN
        self.input_shape = tuple(self.model_params["input_shape"])
        self.params_to_update = self.parameters()
        self.output_channels = self.model_params.output_channels

        self.plotted = False

    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.params_to_update, lr=self.hparam["lr"],
                                     weight_decay=self.hparam["weight_decay"])

        return optimizer

    def general_step(self, batch):
        x = batch
        x = x.to(self.device)
        x_hat = self.model(x)
        loss = nn.MSELoss()(x_hat, x)
        return loss

    def training_step(self, train_batch):
        self.train()
        loss = self.general_step(train_batch)

        return {
            'loss': loss,
        }

    def validation_step(self, val_batch):
        self.eval()
        loss = self.general_step(val_batch)
        return {
            'loss': loss,
        }

    def test_step(self, batch):
        return self.validation_step(batch)

    def general_epoch_end(self, outputs, mode):  ### checked
        # average over all batches aggregated during one epoch
        logger = False if mode == 'test' else True
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        if logger and self.logger is not None:
            self.logger.experiment.add_scalar(f'{mode}/{mode}_loss', avg_loss, self.current_epoch)

        return avg_loss

    def training_epoch_end(self, outputs):
        self.general_epoch_end(outputs, 'train')

    def validation_epoch_end(self, outputs):
        self.general_epoch_end(outputs, 'val')
        self.plotted = False

    def test_epoch_end(self, outputs):
        avg_loss, avg_success = self.general_epoch_end(outputs, 'test')

        return {
            'avg_loss': avg_loss,
            'avg_success': avg_success,
        }

    def infer(self, x: torch.Tensor):
        self.eval()
        if len(x.size()) == 3:
            x = x.unsqueeze(0)

        with torch.no_grad():
           z = self.encoder(x)

        return z