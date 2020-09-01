
from torch.utils.data import Dataset
from typing import Callable

from dcase2020_task4.util.datasets.fn_dataset import FnDataset


class AugmDataset(FnDataset):
	"""
		Apply augment function on first element of every items get in dataset.
	"""
	def __init__(self, dataset: Dataset, augm_fn: Callable):
		super().__init__(dataset, fn=lambda item: (augm_fn(item[0]), item[1]))
