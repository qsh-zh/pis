import numpy as np
import torch as th
from jamtorch.data import num_to_groups
from jamtorch.utils import as_numpy, no_grad_func


@no_grad_func
def generate_traj(sde_model, dt=0.01, t_end=1.0, num_sample=2000):
    # (T, B, D)
    dim = sde_model.data_ndim
    x = th.zeros((num_sample, dim + 1)).float().cuda()
    states = [as_numpy(x[:, :-1])]
    for cur_t in th.arange(0, t_end, dt).cuda():
        f_value = sde_model.f(cur_t, x)
        g_value = sde_model.g(cur_t, x)
        noise = th.randn_like(g_value) * np.sqrt(dt)
        x += f_value * dt + g_value * noise
        states.append(as_numpy(x[:, :-1]))
    return states


# pylint: disable=too-many-locals
@no_grad_func
def generate_samples_loss(
    sde_model,
    nll_target_fn,
    nll_prior_fn,
    dt=0.01,
    t_end=1.0,
    num_sample=2000,
    device="cpu",
):
    x = sde_model.zero(num_sample, device)
    dim = sde_model.data_ndim
    uw_term = 0
    rtn_traj = []
    for cur_t in th.arange(0, t_end, dt).cuda():
        x, cur_uw_term = sde_model.step_with_uw(cur_t, x, dt)
        uw_term += cur_uw_term
        rtn_traj.append(x[:30, :-1].cpu())
    state = x[:, :-1]
    disc_loss, cur_idx = [], 0
    for cur_len_batch in num_to_groups(num_sample, 256):
        disc_loss.append(nll_target_fn(state[cur_idx : cur_idx + cur_len_batch]))
        cur_idx = cur_idx + cur_len_batch
    sample_nll = th.cat(disc_loss)
    prior_nll = nll_prior_fn(state)
    term_loss = sample_nll - prior_nll
    total_loss = x[:, -1] + uw_term + term_loss
    info = {
        "sample/loss": total_loss.mean().item() / dim,
        "sample/sample_nll": sample_nll.mean().item() / dim,
        "sample/prior_nll": prior_nll.mean().item() / dim,
        "sample/term_loss": term_loss.mean().item() / dim,
        "sample/uw_loss": uw_term.mean().item() / dim,
        "sample/reg_loss": x[:, -1].mean().item() / dim,
    }
    return rtn_traj, state, total_loss, info
