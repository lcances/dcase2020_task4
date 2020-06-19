import os.path as osp
import torch

from easydict import EasyDict as edict
from torch import Tensor
from torch.nn.functional import one_hot
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
from typing import List


def sharpen(batch: Tensor, temperature: float, dim: int) -> Tensor:
	""" Sharpen function. Make a distribution more "one-hot" if temperature -> 0. """
	batch = batch ** (1.0 / temperature)
	return normalize(batch, dim=dim)


def sharpen_multi_1(distribution: Tensor, temperature: float, threshold: float) -> Tensor:
	""" Experimental multi-hot sharpening. Currently unused. """
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


def sharpen_multi_2(distribution: Tensor, temperature: float, threshold: float) -> Tensor:
	original_mask = (distribution > threshold).float()
	nb_above = original_mask.sum().long().item()

	if nb_above == 0:
		return distribution

	distribution_expanded = distribution.expand(nb_above, *distribution.shape).clone()
	mask_nums = get_idx_max(original_mask, nb_above)

	mask_nums_expanded = torch.zeros(nb_above, nb_above - 1).long()
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


def sharpen_multi(batch: Tensor, temperature: float, threshold: float) -> Tensor:
	result = batch.clone()
	for i, distribution in enumerate(batch):
		result[i] = sharpen_multi_2(distribution, temperature, threshold)
	return result


def get_idx_max(t: Tensor, nb: int) -> Tensor:
	idx = t.argsort(descending=True)
	return idx[:nb]


def normalize(batch: Tensor, dim: int) -> Tensor:
	""" Return the vector normalized. """
	return batch / batch.norm(p=1, dim=dim).unsqueeze(dim=1)


def same_shuffle(values: List[Tensor]) -> List[Tensor]:
	""" Shuffle each value of values with the same indexes. """
	indices = torch.randperm(len(values[0]))
	for i in range(len(values)):
		values[i] = values[i][indices]
	return values


def binarize_onehot_labels(distributions: Tensor) -> Tensor:
	""" Convert list of distributions vectors to one-hot. """
	indexes = distributions.argmax(dim=1)
	nb_classes = distributions.shape[1]
	bin_labels = one_hot(indexes, nb_classes)
	return bin_labels


def label_to_num(one_hot_vectors: Tensor):
	""" Convert a list of one-hot vectors of size (N, C) to a list of classes numbers of size (N). """
	return one_hot_vectors.argmax(dim=1)


def merge_first_dimension(t: Tensor) -> Tensor:
	""" Reshape tensor of size (M, N, ...) to (M*N, ...). """
	shape = list(t.size())
	shape[1] *= shape[0]
	shape = shape[1:]
	return t.reshape(shape)


def cross_entropy_with_logits(logits: Tensor, targets: Tensor) -> Tensor:
	"""
		Apply softmax on logits and compute cross-entropy with targets.
		Target must be a (batch_size, nb_classes) tensor.
	"""
	pred_x = torch.softmax(logits, dim=1)
	return cross_entropy(pred_x, targets)


def cross_entropy(pred_x: Tensor, targets: Tensor) -> Tensor:
	"""
		Compute cross-entropy with targets.
		Target must be a (batch_size, nb_classes) tensor.
	"""
	return -torch.sum(torch.log(pred_x) * targets, dim=1)


def get_lrs(optim: Optimizer) -> List[float]:
	return [group["lr"] for group in optim.param_groups]


def get_lr(optim: Optimizer, idx: int = 0) -> float:
	return get_lrs(optim)[idx]


def set_lr(optim: Optimizer, new_lr: float):
	for group in optim.param_groups:
		group["lr"] = new_lr


def build_writer(hparams: edict, suffix: str = "") -> SummaryWriter:
	dirname = "%s_%s_%s_%s_%s" % (hparams.dataset_name, hparams.train_name, hparams.model_name, hparams.begin_date, suffix)
	dirpath = osp.join(hparams.logdir, dirname)
	writer = SummaryWriter(log_dir=dirpath, comment=hparams.train_name)
	return writer


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
