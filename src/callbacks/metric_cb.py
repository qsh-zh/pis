# pylint: disable=unused-import
import pytorch_lightning as pl
from pytorch_lightning import Callback

from src.utils.loss_helper import loss2ess_info, loss2logz_info
from src.utils.sampling import generate_samples_loss


# pylint: disable=no-self-use, unnecessary-pass
class VizSampleDist(Callback):
    def __init__(self, every_n, num_sample=6000):
        self.every_n = every_n
        self.num_sample = num_sample

    def on_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if trainer.global_step % self.every_n == 0:
            traj, state, loss, info = generate_samples_loss(
                pl_module.sde_model,
                pl_module.nll_target_fn,
                pl_module.nll_prior_fn,
                pl_module.dt,
                pl_module.t_end,
                self.num_sample,
                device=pl_module.device,
            )

            logz_info = loss2logz_info(loss)
            ess_info = loss2ess_info(loss)
            info.update(logz_info)
            info.update(ess_info)
            pl_module.log_dict(info, on_step=True)
            self.viz_sample(state, trainer, pl_module)
            self.viz_traj(traj, trainer, pl_module)

    def viz_sample(
        self, samples, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        del samples, trainer, pl_module
        pass

    def viz_traj(
        self, traj, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        del traj, trainer, pl_module
        pass
