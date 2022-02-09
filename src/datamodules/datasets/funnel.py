import torch as th
import torch.distributions as D

from .base_set import BaseSet


class FunnelSet(BaseSet):
    def __init__(self, len_data, dim, is_linear=True):
        super().__init__(len_data, is_linear)
        self.data = th.ones(dim, dtype=float).cuda()  # pylint: disable= not-callable
        self.data_ndim = dim

        self.dist_dominant = D.Normal(th.tensor([0.0]).cuda(), th.tensor([1.0]).cuda())
        self.mean_other = th.zeros(dim - 1).float().cuda()
        self.cov_eye = th.eye(dim - 1).float().cuda().view(1, dim - 1, dim - 1)

    def cal_gt_big_z(self):  # pylint: disable=no-self-use
        return 1

    def get_gt_disc(self, x):
        return -self.funner_log_pdf(x)

    def viz_pdf(self, fsave="density.png", lim=3):  # pylint: disable=no-self-use
        pass

    def funner_log_pdf(self, x):
        dominant_x = x[:, 0]
        log_density_dominant = self.dist_dominant.log_prob(dominant_x)  # (B, )

        log_density_other = self._dist_other(dominant_x).log_prob(x[:, 1:])  # (B, )
        return log_density_dominant + log_density_other

    def sample(self, batch_size):
        dominant_x = self.dist_dominant.sample((batch_size,))  # (B,1)
        x_others = self._dist_other(dominant_x).sample()  # (B, dim-1)
        return th.hstack([dominant_x, x_others])

    def _dist_other(self, dominant_x):
        variance_other = th.exp(dominant_x)
        cov_other = variance_other.view(-1, 1, 1) * self.cov_eye
        return D.multivariate_normal.MultivariateNormal(self.mean_other, cov_other)
