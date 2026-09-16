import torch
import torch.nn as nn


def to_2tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x)


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return nn.init.trunc_normal_(
        tensor,
        mean=mean,
        std=std,
        a=a,
        b=b,
    )


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)

        if keep_prob > 0:
            random_tensor.div_(keep_prob)

        return x * random_tensor