import numpy as np

from torch.utils.data import Dataset, Subset
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


def collapse_classes_idx(classes_idx: List[List[int]]) -> List[int]:
	"""
		Resize classes indexes in a single list of indexes.
		@param classes_idx: A list of sublist of int. Every sublist "i" contains the indexes of elements in the classes "i".
		@return: A list of indexes.
	"""
	indexes = []
	for idx in classes_idx:
		indexes += idx
	return indexes


def get_reduced_dataset(dataset: Dataset, nb_classes: int, ratio: float) -> Dataset:
	"""
		Reduce dataset size by ratio but keep the same class distribution.
		@param dataset: The original dataset.
		@param nb_classes: The number of classes in the original dataset.
		@param ratio: The ratio in [0, 1] used to reduce the dataset size.
		@return: The reduced dataset.
	"""
	assert 0.0 <= ratio <= 1.0
	cls_idx_all = get_classes_idx(dataset, nb_classes)
	cls_idx_all = shuffle_classes_idx(cls_idx_all)
	cls_idx_all = reduce_classes_idx(cls_idx_all, ratio)
	indexes = collapse_classes_idx(cls_idx_all)
	return Subset(dataset, indexes)


def get_split_datasets(dataset: Dataset, nb_classes: int, sub_loaders_ratios: List[float]) -> List[Dataset]:
	"""
		Split dataset in several sub-datasets by using a list of ratios.
		Also keep the original class distribution in every sub-dataset.
		@param dataset: The original dataset.
		@param nb_classes: The number of classes in the original dataset.
		@param sub_loaders_ratios: Ratios used to split the dataset. The sum must be 1.
		@return: A list of sub-datasets.
	"""
	cls_idx_all = get_classes_idx(dataset, nb_classes)
	cls_idx_all = shuffle_classes_idx(cls_idx_all)
	idx_split = split_classes_idx(cls_idx_all, sub_loaders_ratios)

	return [Subset(dataset, idx) for idx in idx_split]
