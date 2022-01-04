import torch
from torch import nn
from torchmetrics import ConfusionMatrix
import pytorch_lightning as pl
from abc import ABC

class ClassificationModule(pl.LightningModule, ABC):
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
        self.cm = ConfusionMatrix(num_classes=self.dataset_cfg.n_segments) #, normalize="true")
        # CNN
        self.input_shape = tuple(self.model_params["input_shape"])

        self.params_to_update = []
        self.params_to_update = self.parameters()


    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.params_to_update, lr=self.hparam["lr"],
                                     weight_decay=self.hparam["weight_decay"])

        return optimizer

    def calc_acc(self, y, logits):
        y_hat = nn.Softmax(dim=1)(logits)
        y_hat = torch.argmax(y_hat, dim=1)
        assert y_hat.shape == y.shape, "shape of prediction doesnt match ground truth labels"

        return (y_hat == y).sum() / y.size(0)

    def general_step(self, batch, mode):
        x, y, _ = batch
        output = self(x)

        loss = nn.CrossEntropyLoss()(output, y)
        n_correct = self.calc_acc(y, output)
        if mode == "test":
            self.cm(output, y)
        return loss, n_correct

    def training_step(self, train_batch, batch_idx):
        self.train()
        loss, n_correct = self.general_step(train_batch, "train")
        return {
            'loss': loss,
            'n_correct': n_correct,
        }

    def validation_step(self, val_batch, batch_idx, mode="val"):
        self.eval()
        loss, n_correct = self.general_step(val_batch, mode)
        return {
            'loss': loss.detach().cpu(),
            'n_correct': n_correct.detach().cpu(),
        }

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, mode="test")

    def general_epoch_end(self, outputs, mode):  ### checked
        # average over all batches aggregated during one epoch
        logger = False if mode == 'test' else True
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['n_correct'] for x in outputs]).mean()
        self.log(f'{mode}_loss', avg_loss, logger=logger)
        self.log(f'{mode}_acc', avg_acc, logger=logger)
        return avg_loss, avg_acc

    def training_epoch_end(self, outputs):
        self.general_epoch_end(outputs, 'train')

    def validation_epoch_end(self, outputs):
        self.general_epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs):
        avg_loss, avg_acc = self.general_epoch_end(outputs, 'test')
        return {
            'avg_loss': avg_loss,
            'avg_acc': avg_acc
        }

    def infer(self, images: torch.Tensor):
        self.eval()
        if len(images.size()) == 3:
            images = images.unsqueeze(0)

        with torch.no_grad():
            output = self(images)
        output = nn.Softmax(dim=1)(output)
        probs, indices = torch.max(output, dim=1)
        return probs, indices
