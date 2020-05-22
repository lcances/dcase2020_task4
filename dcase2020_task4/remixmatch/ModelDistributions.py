import torch
from torch import Tensor


class ModelDistributions:
	"""
		Compute mean distributions of a model.
	"""

	def __init__(self, nb_classes: int, max_batches: int = 256, names: list = None):
		if names is None:
			names = ["labeled", "unlabeled"]

		self.nb_classes = nb_classes
		self.max_batches = max_batches
		self.names = names
		self.data = {}

		self.reset()

	def reset(self):
		uniform_distribution = torch.ones(self.max_batches, self.nb_classes).cuda() / self.nb_classes
		self.data = {
			name: [uniform_distribution.clone(), 0] for name in self.names
		}

	def add_batch_pred(self, batch: Tensor, name: str):
		for pred in batch:
			self.add_pred(pred, name)

	def add_pred(self, pred: Tensor, name: str):
		distributions, index = self.data[name]
		distributions[index] = pred
		index = (index + 1) % len(distributions)
		self.data[name][1] = index

	def get_mean_pred(self, name: str) -> Tensor:
		distributions, _ = self.data[name]
		return torch.mean(distributions, dim=0)
