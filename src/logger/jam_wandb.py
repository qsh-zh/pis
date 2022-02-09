import os
import shutil
import socket

from omegaconf import OmegaConf


class WandbUrls:  # pylint: disable=too-few-public-methods
    def __init__(self, url):

        url_hash = url.split("/")[-1]
        project = url.split("/")[-3]
        entity = url.split("/")[-4]

        self.weight_url = url
        self.log_url = "https://app.wandb.ai/{}/{}/runs/{}/logs".format(
            entity, project, url_hash
        )
        self.chart_url = "https://app.wandb.ai/{}/{}/runs/{}".format(
            entity, project, url_hash
        )
        self.overview_url = "https://app.wandb.ai/{}/{}/runs/{}/overview".format(
            entity, project, url_hash
        )
        self.hydra_config_url = (
            "https://app.wandb.ai/{}/{}/runs/{}/files/hydra-config.yaml".format(
                entity, project, url_hash
            )
        )
        self.overrides_url = (
            "https://app.wandb.ai/{}/{}/runs/{}/files/overrides.yaml".format(
                entity, project, url_hash
            )
        )

    # pylint: disable=line-too-long
    def __repr__(self):
        msg = "=================================================== WANDB URLS ===================================================================\n"  # noqa: E501
        for k, v in self.__dict__.items():
            msg += "{}: {}\n".format(k.upper(), v)
        msg += "=================================================================================================================================\n"  # noqa: E501
        return msg

    def to_dict(self):
        return {k.upper(): v for k, v in self.__dict__.items()}


def log_jam(run):
    try:
        from jammy import get_jam_repo_git
    except ImportError:
        return None

    jam_sha, jam_diff = get_jam_repo_git()
    with open("jam_change.patch", "w", encoding="utf8") as f:
        f.write(jam_diff)
    run.save("jam_change.patch")
    return jam_sha


def log_proj(run, proj_path):
    try:
        from jammy.utils import git
    except ImportError:
        return None
    proj_sha, proj_diff = git.log_repo(proj_path)
    with open("proj_change.patch", "w", encoding="utf8") as f:
        f.write(proj_diff)
    run.save("proj_change.patch")
    return proj_sha


def log_hydra(run):
    shutil.copyfile(
        os.path.join(os.getcwd(), ".hydra/config.yaml"),
        os.path.join(os.getcwd(), ".hydra/hydra-config.yaml"),
    )
    run.save(os.path.join(os.getcwd(), ".hydra/hydra-config.yaml"))
    run.save(os.path.join(os.getcwd(), ".hydra/overrides.yaml"))


class JamWandb:
    g_cfg = None
    run = None

    @property
    def cfg(self):
        return JamWandb.g_cfg

    @cfg.setter
    def cfg(self, g_cfg):  # pylint: disable=no-self-use
        JamWandb.g_cfg = g_cfg

    @staticmethod
    def prep_cfg(dump_meta=True):
        if JamWandb.g_cfg is None:
            raise RuntimeError("Set JamWandb g_cfg firstly")
        if JamWandb.run is None:
            raise RuntimeError("Set JamWandb run")
        g_cfg = JamWandb.g_cfg
        run = JamWandb.run
        jam_sha = log_jam(run)
        proj_sha = log_proj(run, g_cfg.work_dir)
        log_hydra(run)
        cfg = {
            "proj_path": g_cfg.work_dir,
            "run_path": os.getcwd(),
            "host": socket.gethostname(),
            "jam_sha": jam_sha,
            "proj_sha": proj_sha,
            **(WandbUrls(run.url).to_dict()),
            "z": OmegaConf.to_container(g_cfg, resolve=True),
        }
        if dump_meta:
            with open("meta.yaml", "w", encoding="utf8") as fp:
                OmegaConf.save(config=OmegaConf.create(cfg), f=fp.name)
        return cfg

    @staticmethod
    def log(*args, **kargs):
        if JamWandb.run is not None:
            raise RuntimeError("wandb is inactive, please launch first.")

        JamWandb.run.log(*args, **kargs)

    @staticmethod
    def finish():
        if JamWandb.run is None:
            return

        if os.path.exists("jam_.log"):
            JamWandb.run.save("jam_.log")

        JamWandb.run.finish()
        JamWandb.run = None
