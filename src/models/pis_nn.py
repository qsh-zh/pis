import copy

import numpy as np
import torch as th
from torch import nn

from src.networks.time_conder import TimeConder
from src.utils.loss_helper import nll_unit_gaussian


def get_reg_fns(fns=None):
    from jammy.utils import imp

    reg_fns = []
    if fns is None:
        return reg_fns

    for _fn in fns:
        reg_fns.append(imp.load_class(_fn))

    return reg_fns


# pylint: disable=function-redefined


class PISNN(nn.Module):  # pylint: disable=abstract-method, too-many-instance-attributes
    def __init__(
        self,
        f_func,
        g_func,
        reg_fns,
        grad_fn=None,
        f_format="f",
        g_coef=np.sqrt(0.2),
        data_shape=2,
        t_end=1.0,
        sde_type="stratonovich",
        noise_type="diagonal",
        nn_clip=1e2,
        lgv_clip=1e2,
    ):  # pylint: disable=too-many-arguments
        super().__init__()
        self.f_func = f_func
        self.g_func = g_func
        self.reg_fns = get_reg_fns(reg_fns)
        self.nreg = len(reg_fns)
        self.sde_type = sde_type
        self.noise_type = noise_type
        self.nn_clip = nn_clip * 1.0
        self.lgv_clip = lgv_clip * 1.0
        self.data_ndim = np.prod(data_shape)
        self.data_shape = (
            tuple(
                [
                    data_shape,
                ]
            )
            if isinstance(data_shape, int)
            else data_shape
        )
        self.grad_fn = grad_fn
        self.g_coef = g_coef

        self.t_end = t_end
        self.select_f(f_format)

    def select_f(self, f_format=None):
        _fn = self.f_func
        if f_format == "f":

            def _fn(t, x):
                return th.clip(self.f_func(t, x), -self.nn_clip, self.nn_clip)

        elif f_format == "t_tnet_grad":
            self.lgv_coef = TimeConder(64, 1, 3)

            def _fn(t, x):
                grad = th.clip(self.grad_fn(x), -self.lgv_clip, self.lgv_clip)
                f = th.clip(self.f_func(t, x), -self.nn_clip, self.nn_clip)
                return f - self.lgv_coef(t) * grad

        elif f_format == "nn_grad":

            def _fn(t, x):
                x_dot = th.clip(self.grad_fn(x), -self.lgv_clip, self.lgv_clip)
                f_x = th.clip(self.f_func(t, x), -self.nn_clip, self.nn_clip)
                return f_x * x_dot

        elif f_format == "comp_grad":
            self.grad_net = copy.deepcopy(self.f_func)

            def _fn(t, x):
                x_dot = th.clip(self.grad_fn(x), -self.lgv_clip, self.lgv_clip)
                f_x = th.clip(self.f_func(t, x), -self.nn_clip, self.nn_clip)
                f_x_dot = th.clip(self.grad_net(t, x_dot), -self.nn_clip, self.nn_clip)
                return f_x + f_x_dot

        else:
            raise RuntimeError

        self.param_fn = _fn

    def f(self, t, state):
        # t: scaler
        # state: Bx(n_state, n_reg)
        class SharedContext:  # pylint: disable=too-few-public-methods
            pass

        x = th.nan_to_num(state[:, : -self.nreg])
        x = x.view(-1, *self.data_shape)
        control = self.param_fn(t, x).flatten(start_dim=1)
        dreg = tuple(reg_fn(x, control, SharedContext) for reg_fn in self.reg_fns)
        return th.cat((control * self.g_coef,) + dreg, dim=1)

    def g(self, t, state):
        origin_g = self.g_func(t, state[:, : -self.nreg]) * self.g_coef
        return th.cat(
            (origin_g, th.zeros((state.shape[0], self.nreg)).to(origin_g)), dim=1
        )

    def zero(self, batch_size, device="cpu"):
        return th.zeros(batch_size, self.data_ndim + self.nreg, device=device)

    def nll_prior(self, state):
        state = state[:, : self.data_ndim]
        return nll_unit_gaussian(state, np.sqrt(self.t_end) * self.g_coef)

    def f_and_g_prod(self, t, y, v):
        v[:, -self.nreg :] = v[:, -self.nreg :] * 0
        return self.f(t, y), v * self.g_coef

    def step_with_uw(self, t, state, dt):
        noise = th.randn_like(state) * np.sqrt(dt)
        f_value, g_prod_noise = self.f_and_g_prod(t, state, noise)
        new_state = state + f_value * dt + g_prod_noise
        uw_term = (f_value[:, :-1] * noise[:, :-1]).sum(dim=1) / self.g_coef
        return new_state, uw_term
