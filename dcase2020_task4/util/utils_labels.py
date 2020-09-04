import numpy as np
import torch

from torch import Tensor
from torch.nn.functional import one_hot
from typing import List, Union


# --- ONE-HOT ---
def nums_to_onehot(nums: Union[np.ndarray, Tensor], nb_classes: int) -> Union[np.ndarray, Tensor]:
	"""
		Convert numbers (or indexes) of classes to one-hot version.
		@param nums: Label of indexes of classes.
		@param nb_classes: The maximum number of distinct classes.
		@return: Label with one-hot vectors
	"""
	if isinstance(nums, Tensor):
		onehots = one_hot(nums, nb_classes)
	else:
		onehots = one_hot(torch.as_tensor(nums), nb_classes).numpy()
	return onehots


def nums_to_smooth_onehot(nums: Union[np.ndarray, Tensor], nb_classes: int, smooth: float) -> Union[np.ndarray, Tensor]:
	"""
		Convert numbers (or indexes) of classes to smooth one-hot version.
		@param nums: Label of indexes of classes.
		@param nb_classes: The maximum number of distinct classes.
		@param smooth: The label smoothing coefficient in [0, 1/nb_classes].
		@return: Label with smooth one-hot vectors
	"""
	onehots = nums_to_onehot(nums, nb_classes)
	return onehot_to_smooth_onehot(onehots, nb_classes, smooth)


def onehot_to_nums(onehots: Tensor) -> Tensor:
	""" Convert a list of one-hot vectors of size (N, C) to a list of classes numbers of size (N). """
	return onehots.argmax(dim=1)


def onehot_to_smooth_onehot(onehots: Union[np.ndarray, Tensor], nb_classes: int, smooth: float) -> Union[np.ndarray, Tensor]:
	""" Smooth one-hot labels with a smoothing coefficient. """
	classes_smoothed = (-onehots + 1.0) * (smooth * (nb_classes - 1)) + (1.0 - smooth) * onehots
	return classes_smoothed


def binarize_pred_to_onehot(pred: Tensor) -> Tensor:
	""" Convert a batch of labels (bsize, label_size) to one-hot by using max(). """
	indexes = pred.argmax(dim=1)
	nb_classes = pred.shape[1]
	onehots = one_hot(indexes, nb_classes)
	return onehots


# --- MULTI-HOT ---
def nums_to_multihot(nums: List[List[int]], nb_classes: int) -> Tensor:
	"""
		Convert a list of numbers (or indexes) of classes to multi-hot version.
		@param nums: List of List of indexes of classes.
		@param nb_classes: The maximum number of classes.
		@return: Label with multi-hot vectors
	"""
	res = torch.zeros((len(nums), nb_classes))
	for i, nums in enumerate(nums):
		res[i] = torch.sum(torch.stack([one_hot(torch.as_tensor(num), nb_classes) for num in nums]), dim=0)
	return res


def multihot_to_nums(multihots: Union[np.ndarray, Tensor], threshold: float = 1.0) -> List[List[int]]:
	"""
		Convert multi-hot vectors to a list of list of classes indexes.
		@param multihots: The multi-hot vectors.
		@param threshold: The threshold used to determine if class is present or not.
		@return: The list of list of classes indexes. Each sub-list can have a different size.
	"""
	res = [
		[j for j, coefficient in enumerate(label) if coefficient >= threshold]
		for i, label in enumerate(multihots)
	]
	return res


def multihot_to_smooth_multihot(multihots: Union[np.ndarray, Tensor], nb_classes: int, smooth: float) -> Union[np.ndarray, Tensor]:
	"""
		Smooth multi-hot labels with a smoothing coefficient.
		@param multihots: Multi-hot vectors.
		@param nb_classes: The maximum number of classes.
		@param smooth: The label smoothing coefficient in [0, 1].
		@return: The smoothed multi-hot vectors.
	"""
	classes_smoothed = (1.0 - smooth) * multihots
	return classes_smoothed
