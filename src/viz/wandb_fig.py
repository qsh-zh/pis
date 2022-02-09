import functools

import matplotlib.pyplot as plt
import wandb

__all__ = ["wandb_plt", "wandb_img"]


def wandb_plt(func):
    @functools.wraps(func)
    def wandb_record(*args, **kwargs):
        fig, caption = func(*args, **kwargs)
        msg = None
        if wandb.run:
            title = (
                caption
                if fig._suptitle is None  # pylint: disable= protected-access
                else fig._suptitle.get_text()  # pylint: disable= protected-access
            )
            msg = {title: wandb.Image(fig, caption=caption)}
            wandb.log(msg)
        plt.close(fig)
        return msg

    return wandb_record


def wandb_img(title, fpath, caption=None):
    if wandb.run:
        msg = {title: wandb.Image(fpath, caption=caption)}
        wandb.log(msg)
