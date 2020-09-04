import numpy as np
import torch

from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Tuple, Union


def normalized(batch: Tensor, dim: int) -> Tensor:
	""" Return the vector normalized. """
	return batch / batch.norm(p=1, dim=dim, keepdim=True)


def same_shuffle(values: List[Tensor]) -> List[Tensor]:
	""" Shuffle each value of values with the same indexes. """
	indices = torch.randperm(len(values[0]))
	for i in range(len(values)):
		values[i] = values[i][indices]
	return values


def merge_first_dimension(t: Tensor) -> Tensor:
	""" Reshape tensor of size (M, N, ...) to (M*N, ...). """
	shape = list(t.size())
	if len(shape) < 2:
		raise RuntimeError("Invalid nb of dimension (%d) for merge_first_dimension. Should have at least 2 dimensions." % len(shape))
	return t.reshape(shape[0] * shape[1], *shape[2:])


def cross_entropy_with_logits(logits: Tensor, targets: Tensor, dim: Union[int, tuple] = 1) -> Tensor:
	"""
		Apply softmax on logits and compute cross-entropy with targets.
		Target must be a (batch_size, nb_classes) tensor.
	"""
	pred_x = torch.softmax(logits, dim=dim)
	return cross_entropy(pred_x, targets, dim)


def cross_entropy(pred: Tensor, targets: Tensor, dim: Union[int, tuple] = 1) -> Tensor:
	"""
		Compute cross-entropy with targets.
		Target must be a (batch_size, nb_classes) tensor.
	"""
	return -torch.sum(torch.log(pred) * targets, dim=dim)


def get_lrs(optim: Optimizer) -> List[float]:
	""" Get the learning rates of an optimizer. """
	return [group["lr"] for group in optim.param_groups]


def get_lr(optim: Optimizer, idx: int = 0) -> float:
	""" Get the learning rate of an optimizer. """
	return get_lrs(optim)[idx]


def set_lr(optim: Optimizer, new_lr: float):
	""" Set the learning rate of an optimizer. """
	for group in optim.param_groups:
		group["lr"] = new_lr


def sq_diff(a: Tensor, b: Tensor) -> Tensor:
	""" Squared root distance (difference) """
	return (a - b) ** 2


def random_rect(
	width: int, height: int, width_range: Tuple[float, float], height_range: Tuple[float, float]
) -> (int, int, int, int):
	"""
		Create a random rectangle inside an area.
		@param width: The maximal width.
		@param height: The maximal height.
		@param width_range: The width ratio range of the rectangle. Ex: (0.1, 0.5) => width is sampled from (0.1 * width, 0.5 * width).
		@param height_range: The height ratio range of the rectangle. Ex: (0, 1) => width is sampled from (0, height).
		@return: The limits (left, right, top, down) of the rectangle created.
	"""

	r_width = np.random.randint(max(1.0, width_range[0] * width), max(2.0, width_range[1] * width))
	r_height = np.random.randint(max(1.0, height_range[0] * height), max(2.0, height_range[1] * height))

	r_left = np.random.randint(0, width - r_width)
	r_top = np.random.randint(0, height - r_height)
	r_right = r_left + r_width
	r_down = r_top + r_height

	return r_left, r_right, r_top, r_down
