import torch

from torch import Tensor
from typing import Callable
from dcase2020_task4.util.utils_match import normalized


class Sharpen(Callable):
	"""
		One-hot sharpening class.
	"""
	def __init__(self, temperature: float):
		self.temperature = temperature

	def __call__(self, batch: Tensor, dim: int) -> Tensor:
		return sharpen(batch, self.temperature, dim)


class SharpenMulti(Callable):
	"""
		Experimental multi-hot sharpening class.
	"""
	def __init__(self, temperature: float, threshold: float):
		self.temperature = temperature
		self.threshold = threshold

	def __call__(self, batch: Tensor, dim: int) -> Tensor:
		return sharpen_multi(batch, self.temperature, self.threshold)


def sharpen(batch: Tensor, temperature: float, dim: int) -> Tensor:
	""" Sharpen function. Make a distribution more "one-hot" if temperature -> 0. """
	batch = batch ** (1.0 / temperature)
	return normalized(batch, dim=dim)


def sharpen_multi(batch: Tensor, temperature: float, threshold: float) -> Tensor:
	"""
		Experimental multi-hot sharpening function.
		@param batch: The batch to sharpen.
		@param temperature: Temperature of the sharpen function.
		@param threshold: Threshold used to determine if a probability must be increased or decreased.
		@return: The batch sharpened.
	"""
	result = batch.clone()
	nb_dim = len(batch.shape)

	if nb_dim == 1:
		return _sharpen_multi_2(batch, temperature, threshold)
	elif nb_dim == 2:
		for i, distribution in enumerate(batch):
			result[i] = _sharpen_multi_2(distribution, temperature, threshold)
	elif nb_dim == 3:
		for i, distribution_i in enumerate(batch):
			for j, distribution_j in enumerate(distribution_i):
				result[i, j] = _sharpen_multi_2(distribution_j, temperature, threshold)
	else:
		raise RuntimeError("Invalid nb_dim %d. (only 1, 2 or 3)" % nb_dim)

	return result


def _sharpen_multi_1(distribution: Tensor, temperature: float, threshold: float) -> Tensor:
	""" Experimental V1 multi-hot sharpening. Currently unused. """
	k = (distribution > threshold).long().sum().item()
	if k < 1:
		return distribution

	sorted_, idx = distribution.sort(descending=True)
	preds, others = sorted_[:k], sorted_[k:]
	preds_idx, others_idx = idx[:k], idx[k:]

	tmp = torch.zeros((len(preds), 1 + len(others)))
	for i, v in enumerate(preds):
		sub_distribution = torch.cat((v.unsqueeze(dim=0), others))
		sub_distribution = sharpen(sub_distribution, temperature, dim=0)
		tmp[i] = sub_distribution

	new_dis = torch.zeros(distribution.size())
	new_dis[preds_idx] = tmp[:, 0].squeeze()
	new_dis[others_idx] = tmp[:, 1:].mean(dim=0)
	return new_dis


def _sharpen_multi_2(distribution: Tensor, temperature: float, threshold: float) -> Tensor:
	""" Experimental V2 multi-hot sharpening. """
	original_mask = (distribution > threshold).float()
	nb_above = original_mask.sum().long().item()

	if nb_above == 0:
		return distribution

	distribution_expanded = distribution.expand(nb_above, *distribution.shape).clone()
	mask_nums = original_mask.argsort(descending=True)[:nb_above]

	mask_nums_expanded = torch.zeros(*distribution.shape[:-1], nb_above, nb_above - 1).long()
	for i in range(nb_above):
		indices = list(range(nb_above))
		indices.remove(i)
		mask_nums_expanded[i] = mask_nums[indices].clone()

	# TODO : rem
	# inverted_mask = -original_mask + 1
	# inverted_nums = get_idx_max(inverted_mask, len(original_mask) - nb_above)
	# mean_below = distribution[inverted_nums].mean()

	for i, (distribution, nums) in enumerate(zip(distribution_expanded, mask_nums_expanded)):
		distribution_expanded[i][nums] = 0.0

	distribution_expanded = sharpen(distribution_expanded, temperature, dim=1)

	result = distribution_expanded.mean(dim=0)
	result[mask_nums] = distribution_expanded.max(dim=1)[0]

	return result
