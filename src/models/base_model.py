from typing import Any

import hydra
import torch
from jammy import hyd_instantiate
from omegaconf import DictConfig
from pytorch_lightning import LightningModule

from src.models.loss import loss_pis
from src.utils.sampling import generate_traj

# pylint: disable=too-many-ancestors,arguments-differ,attribute-defined-outside-init,unused-argument,too-many-instance-attributes, abstract-method


class BaseModel(LightningModule):
    """
    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__()
        self.cfg = cfg
        # TODO:
        self.data_ndim = cfg.data_ndim
        self.dt = cfg.dt
        self.t_end = cfg.t_end
        self.register_buffer("ts", torch.tensor([1e-12, self.t_end]))
        self.instantiate()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

    def instantiate(self):
        f_func = hydra.utils.instantiate(self.cfg.f_func)
        g_func = hydra.utils.instantiate(self.cfg.g_func)
        self.sde_model = hyd_instantiate(
            self.cfg.sde_model, f_func, g_func, t_end=self.t_end
        )
        self.sdeint_fn = hyd_instantiate(self.cfg.sdeint, self.sde_model, dt=self.dt)

    def on_train_start(self) -> None:
        self.sde_model.grad_fn = self.trainer.datamodule.dataset.score
        self.dataset = self.trainer.datamodule.dataset
        self.nll_target_fn = self.trainer.datamodule.dataset.energy
        self.nll_prior_fn = self.sde_model.nll_prior
        if hasattr(self.dataset, "to"):
            self.dataset.to(self.device)

    def training_step(self, batch: Any, batch_idx: int):
        del batch_idx
        batch_size = batch.shape[0]
        _, loss, info = loss_pis(
            self.sdeint_fn,
            self.ts,
            self.nll_target_fn,
            self.nll_prior_fn,
            self.sde_model.zero(batch_size, device=self.device),
            self.sde_model.nreg,
        )
        self.log_dict(info)
        return loss

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(
            params=self.sde_model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

    def sample_n(self, batch_size):
        y0 = self.sde_model.zeros(batch_size)
        ys = self.sdeint_fn(y0, self.ts)
        y1 = ys[-1]

        return y1[:, : self.sde_model.data_ndim]

    def sample_traj(self, batch_size):
        return generate_traj(self.sde_model, self.dt, self.t_end, batch_size)
