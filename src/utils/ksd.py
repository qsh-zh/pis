import torch as th

# pylint: disable=invalid-name,arguments-out-of-order


class RBFKernel:
    """
    A batch implementation
    """

    def __init__(self, beta=1e-2):
        self.beta = beta
        assert self.beta > 0

    def value(self, x, y):  # (B, 1)
        r = ((x - y) ** 2).sum(dim=-1, keepdim=True)
        return th.exp(-self.beta * r)

    def grad_x(self, x, y):  # (B, N)
        r = ((x - y) ** 2).sum(dim=-1, keepdim=True)
        return -2 * self.beta * (x - y) * th.exp(-self.beta * r)

    def grad_y(self, x, y):  # (B,N)
        r = ((x - y) ** 2).sum(dim=-1, keepdim=True)
        return 2 * self.beta * (x - y) * th.exp(-self.beta * r)

    def grad_xy(self, x, y):  # trace (B,N)
        assert len(x) == len(y)
        _, n = x.shape
        r = ((x - y) ** 2).sum(dim=-1, keepdim=True)  # (B,1)
        _y = 2 * self.beta * th.exp(-self.beta * r) * th.ones((1, n)).to(x)  # (B, N)
        _xy = 4 * self.beta ** 2 * (x - y) ** 2 * th.exp(-self.beta * r)
        return _y + _xy


class KSD:
    def __init__(self, score_fn, beta=1e-2):
        self.k_method = RBFKernel(beta)

        self.k = self.k_method.value
        self.grad_kx = self.k_method.grad_x
        self.grad_ky = self.k_method.grad_y
        self.grad_kxy = self.k_method.grad_xy

        self.score_fn = score_fn

    def cal_s(self, x, y):
        log_px = self.score_fn(x)
        log_py = self.score_fn(y)

        p1 = self.k(x, y) * (log_px * log_py).sum(
            dim=-1, keepdims=True
        )  # EQ: k(x,y) * S.T @ S, (B,1)
        p2 = (log_px * self.grad_ky(x, y)).sum(
            dim=-1, keepdims=True
        )  # EQ: Q.T @ \nabla_{x'}k, (B,1)
        p3 = (self.grad_kx(x, y) * log_py).sum(
            dim=-1, keepdims=True
        )  # EQ: \nabla_x k @ Q
        p4 = self.grad_kxy(x, y).sum(dim=-1, keepdims=True)

        return p1 + p2 + p3 + p4

    def __call__(self, x, adjust_beta=False):
        # TODO: adding warning, only support one batch dimension
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        idx = th.combinations(th.arange(batch_size))
        x = x[idx]
        x, y = x[:, 0], x[:, 1]
        if adjust_beta:
            self.k_method.beta = 1.0 / (2 * (x - y) ** 2).mean()
        x, y = x.requires_grad_(True), y.requires_grad_(True)
        return (self.cal_s(x, y) + self.cal_s(y, x)).sum() / (
            batch_size * (batch_size - 1)
        )
