import pytorch_lightning as pl  # pylint: disable=unused-import
from pytorch_lightning import Callback
from torch.optim.lr_scheduler import StepLR


class LinearScheduler(Callback):
    def __init__(self, gamma=0.95):
        self.gamma = gamma
        self.lr_scheduler = None

    def on_pretrain_routine_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        del trainer
        self.lr_scheduler = StepLR(
            pl_module.optimizers(), step_size=1, gamma=self.gamma
        )

    def on_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        del trainer, pl_module
        self.lr_scheduler.step()
