from jammy.utils.hyd import instantiate
from omegaconf.dictconfig import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


# pylint: disable=abstract-method
class DataModule(LightningDataModule):
    """
    Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.cfg = cfg
        self.dataset = instantiate(cfg.dataset)
        self.dims = self.dataset.data_ndim

    def train_dataloader(self):
        return DataLoader(dataset=self.dataset, **self.cfg.dl)

    @classmethod
    def from_args(cls, merge_str=None):
        from omegaconf import OmegaConf

        default = """
        module:
            _target_: src.datamodules.blank_datamodule.DataModule
        dataset:
            _target_: src.datamodules.datasets.ou.OUGaussianMixture
            len_data: 5000
            nmode: 3
            xlim: 1.0
            scale: 0.15
        dl:
            batch_size: 10
            shuffle: true
        ndim: 1
        """
        cfg = OmegaConf.create(default)
        if merge_str is not None:
            merge_cfg = OmegaConf.create(merge_str)
            cfg = OmegaConf.merge(cfg, merge_cfg)
        return cls(cfg)
