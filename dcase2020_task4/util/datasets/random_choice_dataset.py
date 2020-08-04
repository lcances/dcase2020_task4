import numpy as np

from torch.utils.data import Dataset
from typing import List, Optional


class RandomChoiceDataset(Dataset):
	def __init__(self, datasets: List[Dataset], distribution: Optional[List[float]] = None):
		self.datasets = datasets
		self.distribution = distribution

		if self.distribution is None:
			proba = 1.0 / float(len(self.datasets))
			self.distribution = [proba for _ in range(len(self.datasets))]

		self._check_attributes()

	def _check_attributes(self):
		assert len(self.datasets) > 0, "Datasets should not be an empty iterable"

		len_ = len(self.datasets[0])
		for d in self.datasets[1:]:
			assert len(d) == len_, "Datasets must have the same size"

		assert sum(self.distribution) == 1.0, "Distribution sum must be equal to 1.0"

	def __getitem__(self, idx: int):
		dataset_index = np.random.choice(range(len(self.datasets)), p=self.distribution)
		return self.datasets[dataset_index].__getitem__(idx)

	def __len__(self) -> int:
		return len(self.datasets[0])
