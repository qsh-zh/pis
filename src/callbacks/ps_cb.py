# pylint: disable=unused-import
import pytorch_lightning as pl

from src.callbacks.metric_cb import VizSampleDist
from src.viz.ps import viz_kde, viz_sample


class PSSample(VizSampleDist):
    def viz_sample(
        self, samples, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        samples = samples[:, : pl_module.data_ndim]
        viz_sample(samples, "samples", f"sample-{trainer.global_step:04d}.png")
        viz_kde(samples, f"kde-{trainer.global_step:04d}.png")
        if pl_module.data_ndim < 5:
            pl_module.log("ksd", trainer.datamodule.dataset.ksd(samples))
