import matplotlib.pyplot as plt
import seaborn as sns
from jamtorch.utils import as_numpy

from .wandb_fig import wandb_img


def viz_sample(sample, title, fsave, sample_num=50000):
    points = as_numpy(sample)
    fig, axs = plt.subplots(1, 1, figsize=(7, 7))
    axs.set_title(title)
    axs.plot(
        points[:sample_num, 0],
        points[:sample_num, 1],
        linewidth=0,
        marker=".",
        markersize=1,
    )
    axs.set_xlim(-5, 5)
    axs.set_ylim(-5, 5)
    fig.savefig(fsave)
    plt.close(fig)
    wandb_img(title, fsave, fsave)


def viz_kde(points, fname, lim=9.0):
    points = as_numpy(points)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=200)
    sns.kdeplot(
        x=points[:2000, 0], y=points[:2000, 1], cmap="coolwarm", shade=True, ax=ax
    )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axis("off")
    fig.savefig(fname)
    plt.close(fig)
    wandb_img("kde", fname, fname)
