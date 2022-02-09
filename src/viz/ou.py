import matplotlib.pyplot as plt
import numpy as np
import torch as th
from einops import rearrange
from jamtorch.utils import as_numpy, no_grad_func


# pylint: disable=invalid-name
def traj_plot(traj_len, samples, xlabel, ylabel, title="", fsave="img.png"):
    # samples = samples.squeeze().t().cpu()
    # (T,B,D)->(B,T,D)
    samples = rearrange(samples, "t b d -> b t d").cpu()
    inds = np.linspace(0, samples.shape[1], traj_len, endpoint=True, dtype=int)
    samples = samples[:, inds]
    plt.figure()
    for i, sample in enumerate(samples):
        plt.plot(np.range(traj_len), sample.flatten(), marker="x", label=f"sample {i}")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(fsave)
    plt.close()


def dist_plot(samples, nll_target_fn, nll_prior_fn, fname):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    density, bins = np.histogram(samples, 100, density=True)
    query_x = th.linspace(-4.5, 4.5, 100).cuda()
    target_unpdf = th.exp(-nll_target_fn(query_x.view(-1, 1)))
    target_norm_pdf = target_unpdf / th.sum(target_unpdf) / 9 * 100
    prior_unpdf = th.exp(-nll_prior_fn(query_x.view(-1, 1)))
    prior_norm_pdf = prior_unpdf / th.sum(prior_unpdf) / 9 * 100
    ax.plot(bins[1:], density, label="sampled")
    np_x = as_numpy(query_x)
    ax.plot(np_x, as_numpy(target_norm_pdf), label="target")
    ax.plot(np_x, as_numpy(prior_norm_pdf), label="prior")
    ax.set_xlim(np_x[0], np_x[-1])
    ax.set_ylim(0, 1.5 * th.max(target_norm_pdf).item())
    ax.legend()
    fig.savefig(fname)
    plt.close(fig)
    return fname


@no_grad_func
def drfit_surface(model):
    model.cuda()
    xs = th.linspace(-3.0, 3.0, 120).view(-1, 1).cuda()
    ts = th.linspace(0.0, 0.99, 100).cuda()

    values = []
    for cur_t in ts:
        values.append(model.f_func(cur_t, xs))

    values = th.cat(values, dim=1)

    x, t, zz = as_numpy([xs, ts, values])
    tt, xx = np.meshgrid(t, x)
    return tt, xx, zz
