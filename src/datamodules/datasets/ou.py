import matplotlib.pyplot as plt
import torch as th
import torch.distributions as D
from jamtorch.utils import as_numpy
from torch.distributions.mixture_same_family import MixtureSameFamily

from .base_set import BaseSet


class OUGaussianMixture(BaseSet):
    def __init__(self, len_data, nmode=3, xlim=1.0, scale=0.15):
        super().__init__(len_data)
        mix = D.Categorical(th.ones(nmode).cuda())
        xlim = 0.01 if nmode == 1 else xlim
        comp = D.Normal(
            th.linspace(-xlim, xlim, nmode).cuda(), th.ones(nmode).cuda() * scale
        )
        self.gmm = MixtureSameFamily(mix, comp)
        self.data = th.tensor([0.0])  # pylint: disable= not-callable

    def get_gt_disc(self, x):
        return -self.gmm.log_prob(x).flatten()

    def sample(self, batch_size):
        return self.gmm.sample((batch_size,))

    def viz_pdf(self, fsave="ou-density.png"):
        x = th.linspace(-10, 10, 100).cuda()
        density = self.unnorm_pdf(x)
        x, pdf = as_numpy([x, density])
        fig, axs = plt.subplots(1, 1, figsize=(1 * 7, 1 * 7))
        axs.plot(x, pdf)
        fig.savefig(fsave)
        plt.close(fig)

    def __getitem__(self, idx):
        del idx
        return self.data[0]
