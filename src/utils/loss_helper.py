import numpy as np
import torch as th


def nll_unit_gaussian(data, sigma=1.0):
    data = data.view(data.shape[0], -1)
    loss = 0.5 * np.log(2 * np.pi) + np.log(sigma) + 0.5 * data * data / sigma ** 2
    return th.sum(th.flatten(loss, start_dim=1), -1)


def linear_intepolate_energy(origin_energy_fn, x, weight_gauss=1.0):
    origin_energy = origin_energy_fn(x)

    gaussian_energy = nll_unit_gaussian(x)
    assert origin_energy.shape == gaussian_energy.shape

    return weight_gauss * gaussian_energy + (1.0 - weight_gauss) * origin_energy


def loss2logz_info(loss):
    # loss: accumulated loss for trajs seperately
    log_weight = -loss + loss.mean()
    unnormal_weight = th.exp(log_weight)
    weight = unnormal_weight / unnormal_weight.sum()
    half_unnormal_weight = th.exp(log_weight / 2)
    half_weigh = half_unnormal_weight / half_unnormal_weight.sum()
    return {
        "logz/loss_lower_bound": -loss.mean(),
        "logz/loss_upper_bound": th.sum(-weight * loss),
        "logz/loss_half_bound": th.sum(-half_weigh * loss),
        "logz/loss_unbiased": th.log(th.mean(th.exp(log_weight))) - loss.mean(),
    }


def loss2ess_info(loss):
    log_weight = -loss + loss.mean()
    unnormal_weight = th.exp(log_weight)
    weight = unnormal_weight / unnormal_weight.sum()
    return {"ess": 1.0 / (weight * weight).sum() / len(weight)}
