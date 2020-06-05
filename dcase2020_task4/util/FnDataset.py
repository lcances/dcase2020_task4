
from torch.utils.data import Dataset
from typing import Callable


class FnDataset(Dataset):
	def __init__(self, dataset: Dataset, fn: Callable):
		super().__init__()
		self.dataset = dataset
		self.fn = fn

	def __getitem__(self, idx: int):
		return self.fn(self.dataset[idx])

	def __len__(self) -> int:
		return len(self.dataset)
