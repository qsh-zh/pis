import os.path as osp
import pathlib

import torch as th

from .base_set import BaseSet
from .cox_utils import Cox


class CoxDist(BaseSet):
    def __init__(self, len_data, dim, is_linear=True):
        fcsv = osp.join(pathlib.Path(__file__).parent.resolve(), "df_pines.csv")
        self.cox = Cox(fcsv, 40, use_whitened=False)

        super().__init__(len_data, is_linear)
        self.data = th.ones(dim, dtype=float).cuda()  # pylint: disable= not-callable
        self.data_ndim = dim

    def get_gt_disc(self, x):
        return -self.cox.evaluate_log_density(x)
