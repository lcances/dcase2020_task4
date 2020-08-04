import torch

from torch.nn import Module
from torch.nn.functional import softplus


class Mish(Module):
	def __call__(self, x):
		return mish(x)


def mish(x):
	return x * torch.tanh(softplus(x))
