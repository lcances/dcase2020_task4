
from torch.utils.data import DataLoader
from typing import Callable


class FnDataset(DataLoader):
	def __init__(self, *args, fn: Callable, **kwargs):
		super().__init__(*args, **kwargs)
		self.fn = fn

	def __iter__(self):
		for item in super().__iter__():
			yield self.fn(item)
