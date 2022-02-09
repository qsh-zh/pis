import torch as th


# pylint: disable=too-few-public-methods
class IdentityOne:
    def __call__(self, t, y):
        del t
        return th.ones_like(y)
