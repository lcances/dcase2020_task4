import numpy as np

from torch.utils.data import Dataset


def get_classes_idx(dataset: Dataset, nb_classes: int) -> list:
	result = [[] for _ in range(nb_classes)]
	for i in range(len(dataset)):
		data, label = dataset[i]
		result[label].append(i)
	return result


def shuffle_classes_idx(classes_idx: list) -> list:
	result = []
	for indexes in classes_idx:
		np.random.shuffle(indexes)
		result.append(indexes)
	return result


def reduce_classes_idx(classes_idx: list, ratio: float) -> list:
	result = []
	for indexes in classes_idx:
		# Get a small part of the class
		idx_dataset_end = int(len(indexes) * ratio)
		indexes = indexes[:idx_dataset_end]
		result.append(indexes)
	return result


def split_classes_idx(classes_idx: list, ratios: list) -> list:
	result = [[] for _ in range(len(ratios))]
	for indexes in classes_idx:
		begins = [int(sum(ratios[:i]) * len(indexes)) for i in range(len(ratios))]
		ends = begins[1:] + [None]
		indexes_split = [indexes[i:j] for i, j in zip(begins, ends)]

		for i in range(len(ratios)):
			result[i] += indexes_split[i]
	return result
