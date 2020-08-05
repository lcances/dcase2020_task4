import numpy as np
import torch

from torch import Tensor
from torch.nn.functional import one_hot
from typing import List, Union


def nums_to_onehot(nums: Union[np.ndarray, Tensor], nb_classes: int) -> Union[np.ndarray, Tensor]:
	if isinstance(nums, Tensor):
		onehots = one_hot(nums, nb_classes)
	else:
		onehots = one_hot(torch.as_tensor(nums), nb_classes).numpy()
	return onehots


def nums_to_multihot(nums: List[List[int]], nb_classes: int) -> Tensor:
	res = torch.zeros((len(nums), nb_classes))
	for i, nums in enumerate(nums):
		res[i] = torch.sum(torch.stack([one_hot(torch.as_tensor(num), nb_classes) for num in nums]), dim=0)
	return res


def nums_to_smooth_onehot(nums: Union[np.ndarray, Tensor], nb_classes: int, smooth: float) -> Union[np.ndarray, Tensor]:
	onehots = nums_to_onehot(nums, nb_classes)
	return onehot_to_smooth_onehot(onehots, nb_classes, smooth)


def onehot_to_nums(onehots: Tensor) -> Tensor:
	""" Convert a list of one-hot vectors of size (N, C) to a list of classes numbers of size (N). """
	return onehots.argmax(dim=1)


def onehot_to_smooth_onehot(onehots: Union[np.ndarray, Tensor], nb_classes: int, smooth: float) -> Union[np.ndarray, Tensor]:
	classes_smoothed = (-onehots + 1.0) * (smooth * (nb_classes - 1)) + (1.0 - smooth) * onehots
	return classes_smoothed


def multihot_to_nums(multihots: Union[np.ndarray, Tensor], threshold: float = 1.0) -> List[List[int]]:
	res = [
		[j for j, coefficient in enumerate(label) if coefficient >= threshold]
		for i, label in enumerate(multihots)
	]
	return res


def multihot_to_smooth_multihot(multihots: Union[np.ndarray, Tensor], nb_classes: int, smooth: float) -> Union[np.ndarray, Tensor]:
	classes_smoothed = (1.0 - smooth) * multihots
	return classes_smoothed


def binarize_pred_to_onehot(pred: Tensor) -> Tensor:
	""" Convert a batch of labels (bsize, label_size) to one-hot by using max(). """
	indexes = pred.argmax(dim=1)
	nb_classes = pred.shape[1]
	onehots = one_hot(indexes, nb_classes)
	return onehots
