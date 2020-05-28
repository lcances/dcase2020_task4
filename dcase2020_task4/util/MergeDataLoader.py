
from torch import Tensor
from torch.utils.data import DataLoader
from typing import List


class MergeDataLoader:
	"""
		TODO: doc
	"""

	def __init__(self, loaders: list):
		self._loaders = loaders

		# Check loaders size for avoid exception in __iter__
		for loader in self._loaders:
			if len(loader) == 0:
				raise RuntimeError("A sub-DataLoader is empty.")

	def __iter__(self) -> List[Tensor]:
		iters = [iter(loader) for loader in self._loaders]

		for _ in range(len(self)):
			items = []
			for i, _ in enumerate(iters):
				try:
					item = next(iters[i])
				except StopIteration:
					iters[i] = iter(self._loaders[i])
					item = next(iters[i])
				items += list(item)

			yield items

	def __len__(self) -> int:
		return max([len(loader) for loader in self._loaders])

	@property
	def loader_supervised(self) -> DataLoader:
		return self._loaders[0]

	@property
	def loader_unsupervised(self) -> DataLoader:
		return self._loaders[1]
