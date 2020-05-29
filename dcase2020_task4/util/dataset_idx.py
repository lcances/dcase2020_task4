import numpy as np

from torch.utils.data import Dataset
from typing import List


def get_classes_idx(dataset: Dataset, nb_classes: int) -> List[List[int]]:
	"""
		Get class indexes from a standard dataset with index of class as label.
	"""
	result = [[] for _ in range(nb_classes)]
	for i in range(len(dataset)):
		data, label = dataset[i]
		result[label].append(i)
	return result


def shuffle_classes_idx(classes_idx: List[List[int]]) -> List[List[int]]:
	"""
		Shuffle each class indexes.
	"""
	result = []
	for indexes in classes_idx:
		np.random.shuffle(indexes)
		result.append(indexes)
	return result


def reduce_classes_idx(classes_idx: List[List[int]], ratio: float) -> List[List[int]]:
	"""
		Reduce class indexes by a ratio.
	"""
	result = []
	for indexes in classes_idx:
		idx_dataset_end = max(int(len(indexes) * ratio), 0)
		indexes = indexes[:idx_dataset_end]
		result.append(indexes)
	return result


def split_classes_idx(classes_idx: List[List[int]], ratios: List[float]) -> List[List[int]]:
	"""
		Split class indexes and merge them for each ratio.

		Ex:
			input:  classes_idx = [[1, 2], [3, 4], [5, 6]], ratios = [0.5, 0.5]
			output: [[1, 3, 5], [2, 4, 6]]
	"""
	result = [[] for _ in range(len(ratios))]
	for indexes in classes_idx:
		begins = [int(sum(ratios[:i]) * len(indexes)) for i in range(len(ratios))]
		ends = begins[1:] + [None]
		indexes_split = [indexes[i:j] for i, j in zip(begins, ends)]

		for i in range(len(ratios)):
			result[i] += indexes_split[i]
	return result
