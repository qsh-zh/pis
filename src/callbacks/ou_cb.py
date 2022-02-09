# pylint: disable=unused-import
import pytorch_lightning as pl
from jamtorch.utils import as_numpy
from pytorch_lightning import Callback

from src.callbacks.metric_cb import VizSampleDist
from src.viz.ou import dist_plot, traj_plot
from src.viz.wandb_fig import wandb_img


class OUSample(VizSampleDist):
    def viz_sample(
        self, samples, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        samples = samples[:, : pl_module.data_ndim]
        fname = f"x-{trainer.global_step:04d}.png"
        dist_plot(
            as_numpy(samples),
            pl_module.nll_target_fn,
            pl_module.nll_prior_fn,
            fname,
        )
        wandb_img("x", fname, trainer.global_step)
        if pl_module.data_ndim < 5:
            pl_module.log("ksd", trainer.datamodule.dataset.ksd(samples))


class HMTOUSample(OUSample):
    def viz_sample(
        self, samples, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        samples_x = samples[:, : pl_module.data_ndim]
        fname = f"x-{trainer.global_step:04d}.png"
        dist_plot(
            as_numpy(samples_x),
            pl_module.nll_target_x,
            pl_module.sde_model.nll_prior_x,
            fname,
        )
        wandb_img("x", fname, trainer.global_step)
        if pl_module.data_ndim < 5:
            pl_module.log("ksd", trainer.datamodule.dataset.ksd(samples_x))

        samples_v = samples[:, pl_module.data_ndim : 2 * pl_module.data_ndim]
        fname = f"vel-{trainer.global_step:04d}.png"
        dist_plot(
            as_numpy(samples_v),
            pl_module.nll_target_v,
            pl_module.nll_target_v,
            fname,
        )
        wandb_img("vel", fname, trainer.global_step)


class VizSampleTraj(Callback):
    def __init__(self, every_n, traj_num, viz_traj_len):
        self.every_n = every_n
        self.traj_num = traj_num
        self.viz_traj_len = viz_traj_len

    def on_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if trainer.global_step % self.every_n == 0:
            traj = pl_module.sample_traj(self.traj_num)
            traj_plot(
                self.viz_traj_len,
                traj,
                xlabel="$t$",
                ylabel="$Y_t$",
                title=f"Iter {trainer.global_step:04d}",
                fsave=f"traj-{trainer.global_step:04d}.png",
            )
            wandb_img(
                "traj", f"traj-{trainer.global_step:04d}.png", trainer.global_step
            )
