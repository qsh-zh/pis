import torch as th
from jammy.utils.imp import load_class

from .base_set import BaseSet


class ImgSet(BaseSet):
    def __init__(self, dist_class, len_data, is_linear=True):
        super().__init__(len_data, is_linear=is_linear)
        self.net = load_class("jamdist.distribution." + dist_class)()
        self.data_ndim = self.net.data_ndim
        self.data_shape = self.net.data_shape
        self.data = th.randn(1, 1)

    def to(self, device="cpu"):
        self.net.to(device)

    def get_gt_disc(self, x):
        # TODO: fix the coef in the jamdist package
        nll_ebm = self.net.forward(x.view(-1, *self.data_shape)).flatten() * 35555
        return nll_ebm

    # def get_gt_disc(self, x):
    #     nll_ebm = -self.net.forward(x.view(-1, *self.data_shape)).flatten() * self.ndim
    #     nll_sharp_gaussian_pixel = (F.relu(th.abs(x) - 1)) ** 2 * 100
    #     nll_gaussian = th.sum(
    #         th.flatten(nll_sharp_gaussian_pixel, start_dim=1), dim=-1
    #     ).flatten()
    #     return nll_ebm + nll_gaussian
