import abc

import torch as th
from torch.utils.data import Dataset

from src.utils.ksd import KSD
from src.utils.loss_helper import linear_intepolate_energy, nll_unit_gaussian


class BaseSet(abc.ABC, Dataset):
    def __init__(self, len_data, is_linear=True):
        self.num_sample = len_data
        self.annealing_temp = 0.0 if is_linear else 1.0
        self.sample_energy_fn = (
            self.linear_intepolate if is_linear else self.temp_intepolate
        )

        self.data = None
        self.data_ndim = None
        self.worker = KSD(self.score, beta=0.2)
        self._gt_ksd = None

    @property
    def temp(self):
        return self.annealing_temp

    @temp.setter
    def temp(self, value):
        self.annealing_temp = value

    @abc.abstractmethod
    def get_gt_disc(self, x):
        return

    def energy(self, x):
        return self.sample_energy_fn(x).flatten()

    def unnorm_pdf(self, x):
        return th.exp(-self.energy(x))

    def hmt_energy(self, x):
        x, v = th.split(x, 2, dim=1)
        neg_log_p_x = self.sample_energy_fn(x)
        neg_log_p_v = nll_unit_gaussian(v)
        return neg_log_p_x + neg_log_p_v

    @property
    def ndim(self):
        return self.data_ndim

    def sample(self, batch_size):  # pylint: disable=no-self-use
        del batch_size
        raise NotImplementedError

    def temp_intepolate(self, x):
        return self.get_gt_disc(x) / self.annealing_temp

    def linear_intepolate(self, x):
        weight_gauss = self.annealing_temp
        return linear_intepolate_energy(self.get_gt_disc, x, weight_gauss=weight_gauss)

    def score(self, x):
        with th.no_grad():
            copy_x = x.detach().clone()
            copy_x.requires_grad = True
            with th.enable_grad():
                # TODO: should it be _gt_disc or _disc
                self.energy(copy_x).sum().backward()
                lgv_data = copy_x.grad.data
            return lgv_data

    def __len__(self):
        return self.num_sample

    def __getitem__(self, idx):
        return self.data

    def ksd(self, points):
        with th.no_grad():
            cur_ksd = self.gt_ksd()
        return self.worker(points) - cur_ksd

    def gt_ksd(self):
        if self._gt_ksd is None:
            with th.no_grad():
                self._gt_ksd = self.worker(
                    self.sample(5000).view(5000, -1), adjust_beta=True
                )
        return self._gt_ksd
