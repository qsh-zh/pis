import torch as th

# pylint: disable=too-many-arguments


def quad_reg(x, dx, context):
    del x, context
    dx = dx.view(dx.shape[0], -1)
    return 0.5 * dx.pow(2).sum(dim=-1, keepdim=True)


def loss_pis(sdeint_fn, ts, nll_target_fn, nll_prior_fn, y0, n_reg):
    ys = sdeint_fn(y0, ts)
    y1 = ys[-1]
    dim = y1.shape[1]

    reg_loss = y1[:, -n_reg].mean() / dim
    state = th.nan_to_num(y1[:, :-n_reg])
    sample_nll = nll_target_fn(state).mean() / dim
    prior_nll = nll_prior_fn(state).mean() / dim
    term_loss = sample_nll - prior_nll
    loss = reg_loss + term_loss
    return (
        state,
        loss,
        {
            "loss": loss,
            "reg_loss": reg_loss,
            "prior_nll": prior_nll,
            "sample_nll": sample_nll,
            "term_loss": term_loss,
        },
    )
