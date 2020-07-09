import torch

from torch import Tensor
from torch.nn.functional import one_hot
from torch.optim.optimizer import Optimizer
from typing import List, Union


def normalized(batch: Tensor, dim: int) -> Tensor:
	""" Return the vector normalized. """
	return batch / batch.norm(p=1, dim=dim, keepdim=True)


def same_shuffle(values: List[Tensor]) -> List[Tensor]:
	""" Shuffle each value of values with the same indexes. """
	indices = torch.randperm(len(values[0]))
	for i in range(len(values)):
		values[i] = values[i][indices]
	return values


def binarize_onehot_labels(batch: Tensor) -> Tensor:
	""" Convert a batch of labels (bsize, label_size) to one-hot by using max(). """
	indexes = batch.argmax(dim=1)
	nb_classes = batch.shape[1]
	bin_labels = one_hot(indexes, nb_classes)
	return bin_labels


def label_to_num(one_hot_vectors: Tensor):
	""" Convert a list of one-hot vectors of size (N, C) to a list of classes numbers of size (N). """
	return one_hot_vectors.argmax(dim=1)


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
	return [group["lr"] for group in optim.param_groups]


def get_lr(optim: Optimizer, idx: int = 0) -> float:
	return get_lrs(optim)[idx]


def set_lr(optim: Optimizer, new_lr: float):
	for group in optim.param_groups:
		group["lr"] = new_lr


def multi_hot(labels_nums: List[List[int]], nb_classes: int) -> Tensor:
	""" TODO : test this fn """
	res = torch.zeros((len(labels_nums), nb_classes))
	for i, nums in enumerate(labels_nums):
		res[i] = torch.sum(torch.stack([one_hot(num) for num in nums]), dim=0)
	return res


def multilabel_to_num(labels: Tensor) -> List[List[int]]:
	""" TODO : test this fn """
	res = [[] for _ in range(len(labels))]
	for i, label in enumerate(labels):
		for j, bin in enumerate(label):
			if bin == 1.0:
				res[i].append(j)
	return res
