from torch.utils.data import DataLoader
from typing import Callable


class FnDataLoader(DataLoader):
	def __init__(self, fn: Callable, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.fn = fn

	def __iter__(self):
		for x, y in super().__iter__():
			yield self.fn(x, y)
