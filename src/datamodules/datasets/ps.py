import abc
import os.path as osp
import pathlib

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.distributions as D
from jamtorch.utils import as_numpy
from torch.distributions.mixture_same_family import MixtureSameFamily

from .base_set import BaseSet
from .img_tools import ImageEnergy, prepare_image

# pylint: disable=global-statement, global-variable-not-assigned

g_fns = {}


def register(func):
    global g_fns
    g_fns[func.__name__] = func
    return func


@register
def checkerboard(x):
    x_pos_mod = th.div(2 * x[:, 0], 1, rounding_mode="floor")
    y_pos_mod = th.div(2 * x[:, 1], 1, rounding_mode="floor")
    # dx, dy = th.abs(x[:, 0]) - 2, th.abs(x[:,1]) - 2
    p_dist = th.abs(x) - 1
    sign_d = th.clip(
        th.norm(th.clip(p_dist, 0.0), dim=1) + th.clip(th.max(p_dist, dim=1)[0], max=0),
        0,
    )
    value = (x_pos_mod + y_pos_mod) % 2 * 1e6 + sign_d ** 2 * 1e5
    return value


@register
def strip(x):
    x_pos_mod = th.div(2 * x[:, 0], 1, rounding_mode="floor")
    value = x_pos_mod % 2 * 1e6 + ((x_pos_mod < -2) | (x_pos_mod >= 2)) * 1e6
    return th.clip(value, 0, 1e6)


# pylint: disable=attribute-defined-outside-init


class Base2DSet(BaseSet, abc.ABC):  # pylint: disable=abstract-method
    """4x4, [-1,1]"""

    def __init__(self, len_data, is_linear=True):
        super().__init__(len_data, is_linear)
        self.data = th.tensor([0.0, 0.0])  # pylint: disable= not-callable
        self.data_ndim = 2

    def viz_pdf(self, fsave="checkboard-density.png", lim=6):
        x = th.linspace(-lim, lim, 100).cuda()
        xx, yy = th.meshgrid(x, x)
        points = th.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)
        # un_pdf = th.exp(-self.get_gt_disc(points))
        un_pdf = self.unnorm_pdf(points)

        fig, axs = plt.subplots(1, 1, figsize=(1 * 7, 1 * 7))
        axs.imshow(as_numpy(un_pdf.view(100, 100)))
        fig.savefig(fsave)
        plt.close(fig)

    def cal_gt_big_z(self):
        x = th.linspace(-10, 10, 500).cuda()
        xx, yy = th.meshgrid(x, x)
        points = th.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)
        un_pdf = self.unnorm_pdf(points)
        pdf = un_pdf / un_pdf.sum()
        return (pdf * un_pdf).sum()


class FnPs(Base2DSet):
    def __init__(self, len_data, is_linear=True, fn_str="checkerboard"):
        global g_fns
        self.fn = g_fns[fn_str]
        super().__init__(len_data, is_linear)

    def get_gt_disc(self, x):
        return self.fn(x)


class MG2D(Base2DSet):
    def __init__(self, len_data, is_linear=True, nmode=3, xlim=3.0, scale=0.15):
        mix = D.Categorical(th.ones(nmode).cuda())
        angles = np.linspace(0, 2 * 3.14, nmode, endpoint=False)
        poses = xlim * np.stack([np.cos(angles), np.sin(angles)]).T
        poses = th.from_numpy(poses).cuda()
        comp = D.Independent(
            D.Normal(poses, th.ones(size=(nmode, 2)).cuda() * scale * xlim), 1
        )

        self.gmm = MixtureSameFamily(mix, comp)

        super().__init__(len_data, is_linear)

    def get_gt_disc(self, x):
        return -self.gmm.log_prob(x)

    def sample(self, batch_size):
        return self.gmm.sample((batch_size,))


class Fun(Base2DSet):
    def __init__(self, len_data, is_linear=True):
        mix = D.Categorical(th.ones(9).cuda())
        xx, yy = np.mgrid[-5:5:3j, -5:5:3j]
        poses = th.from_numpy(np.vstack([xx.flatten(), yy.flatten()]).T).cuda()
        comp = D.Independent(
            D.Normal(poses, 1.0 * th.ones(size=(9, 2)).cuda() * np.sqrt(0.3)), 1
        )
        self.gmm = MixtureSameFamily(mix, comp)

        super().__init__(len_data, is_linear)

    def get_gt_disc(self, x):
        return -self.gmm.log_prob(x)

    def sample(self, batch_size):
        return self.gmm.sample((batch_size,))


class Rings(Base2DSet):
    def __init__(
        self, len_data, lower=1.0, upper=5.0, num_r=3, r_var=0.01, is_linear=True
    ):
        self.r_centers = th.linspace(lower, upper, num_r).cuda()
        self.var = r_var

        super().__init__(len_data, is_linear)

    def get_gt_disc(self, x):
        radius = th.norm(x, p=2, dim=1).view(-1, 1)
        return th.min((radius - self.r_centers) ** 2, dim=1).values / self.var

    def linear_intepolate(self, x):
        neg_log_p = self.get_gt_disc(x)

        # calculate gaussian part
        x = x.flatten(start_dim=1)
        x_res = x - th.mean(x, dim=0, keepdim=True)
        neg_log_gaussian = 0.5 * th.sum(x_res * x_res, dim=1) / 5

        assert neg_log_p.shape == neg_log_gaussian.shape

        return self.temp * neg_log_gaussian + (1 - self.temp) * neg_log_p


class Chlg2D(Base2DSet):
    def __init__(self, len_data, is_linear=True):
        mix = D.Categorical(th.ones(3).cuda())
        mean = th.tensor([[0.9, 0.0], [-0.75, 0.0], [0.6, 0.9]]).cuda() / 0.3
        cov = (
            th.tensor(
                [
                    [[0.063, 0.0], [0.0, 0.0045]],
                    [[0.063, 0.0], [0.0, 0.0045]],
                    [[0.09, 0.085], [0.085, 0.09]],
                ]
            ).cuda()
            / 0.09
        )
        comp = D.Independent(D.multivariate_normal.MultivariateNormal(mean, cov), 0)

        self.gmm = MixtureSameFamily(mix, comp)
        super().__init__(len_data, is_linear)

    def get_gt_disc(self, x):
        return -th.log(
            (
                th.exp(self.gmm.log_prob(x))
                + th.exp(self.gmm.log_prob(x.flip(dims=(-1,))))
            )
            / 2.0
        )

    def sample(self, batch_size):
        return self.gmm.sample((batch_size,))


class ImgPs(Base2DSet):
    def __init__(self, len_data, is_linear=True):
        fimg = osp.join(pathlib.Path(__file__).parent.resolve(), "labrador.jpg")
        # fimg = osp.join(pathlib.Path(__file__).parent.resolve(), "smiley.jpg")
        img = mpimg.imread(fimg)
        _, img_energy = prepare_image(
            img,
            crop=(10, 710, 240, 940),
            white_cutoff=225,
            gauss_sigma=3,
            background=0.01,
        )
        self._energy_fn = ImageEnergy(
            img_energy[::-1].copy(), mean=[350, 350], scale=[300, 300]
        )
        super().__init__(len_data, is_linear)

    def get_gt_disc(self, x):
        return self._energy_fn.energy(x.cpu()).cuda().flatten()
