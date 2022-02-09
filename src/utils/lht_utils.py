import contextlib
import functools
import logging
import sys
import warnings
from typing import List, Sequence

import pytorch_lightning as pl
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only

try:
    from jammy import link_hyd_run
except ImportError:

    def link_hyd_run(dst_fname=".latest_exp", proj_path=None):
        import os

        import hydra

        exp_folder = os.getcwd()
        if proj_path is None:
            proj_path = hydra.utils.get_original_cwd()
        target_path = os.path.join(proj_path, dst_fname)
        try:
            os.symlink(exp_folder, target_path)
        except FileExistsError:
            os.unlink(target_path)
            os.symlink(exp_folder, target_path)
        g_logger.info(f"{exp_folder} ==>> {target_path}")


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    _logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(_logger, level, rank_zero_only(getattr(_logger, level)))

    return _logger


g_logger = get_logger(__name__)


def extras(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - forcing debug friendly configuration
    - verifying experiment name is set when running in experiment mode

    Modifies DictConfig in place.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    log = get_logger(__name__)

    # quick link
    rank_zero_only(link_hyd_run)()
    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # verify experiment name is set when running in experiment mode
    if config.get("experiment_mode") and not config.get("name"):
        log.info(
            "Running in experiment mode without the experiment name specified! "
            "Use `python run.py mode=exp name=experiment_name`"
        )
        log.info("Exiting...")
        exit()

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    # debuggers don't like GPUs and multiprocessing
    if config.trainer.get("fast_dev_run"):
        log.info(
            "Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>"
        )
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "trainer",
        "model",
        "datamodule",
        "callbacks",
        "logger",
        "test_after_training",
        "seed",
        "name",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.log", "w", encoding="utf-8") as fp:
        rich.print(tree, file=fp)


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.

    Additionaly saves:
        - number of model parameters
    """
    del datamodule, callbacks, logger

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["trainer"] = config["trainer"]
    hparams["model"] = config["model"]
    hparams["datamodule"] = config["datamodule"]

    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)


def auto_gpu(config: DictConfig):
    if config.agpu:
        if getattr(config.trainer, "gpus", 0) > 0:
            try:
                from jammy.cli.gpu_sc import get_gpu_by_utils
            except ImportError:
                g_logger.warning("auto_gpu fails")
                g_logger.warning("auto_gpu needs jammy support")
                return

            best_id = get_gpu_by_utils(
                num_gpus=getattr(config.trainer, "gpus", 1), sleep_sec=20
            )
            config.trainer.gpus = best_id
            g_logger.warning(f"auto_gpu select device {best_id}")


def finish(  # pylint: disable= unused-argument
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for clogger in logger:
        from src.logger.wandb_logger import WandbLogger

        if isinstance(clogger, WandbLogger):
            from src.logger.jam_wandb import JamWandb

            JamWandb.finish()


def _custom_exception_hook(f_type, value, tb):  # pylint: disable=invalid-name
    if hasattr(sys, "ps1") or not sys.stderr.isatty():
        # we are in interactive mode or we don't have a tty-like
        # device, so we call the default hook
        sys.__excepthook__(f_type, value, tb)
    else:
        import traceback

        import ipdb

        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(f_type, value, tb)
        # ...then start the debugger in post-mortem mode.
        ipdb.post_mortem(tb)


def hook_exception_ipdb():
    if not hasattr(_custom_exception_hook, "origin_hook"):
        _custom_exception_hook.origin_hook = sys.excepthook
        sys.excepthook = _custom_exception_hook


def unhook_exception_ipdb():
    assert hasattr(_custom_exception_hook, "origin_hook")
    sys.excepthook = _custom_exception_hook.origin_hook


@contextlib.contextmanager
def exception_hook(enable=True):
    if enable:
        hook_exception_ipdb()
        yield
        unhook_exception_ipdb()
    else:
        yield


def decorate_exception_hook(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        with exception_hook():
            return func(*args, **kwargs)

    return wrapped
