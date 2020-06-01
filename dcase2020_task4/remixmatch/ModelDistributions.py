import torch
from torch import Tensor
from typing import Optional, List


class ModelDistributions:
	"""
		Compute mean output distributions of a model.
	"""

	def __init__(
		self, history_size: int, nb_classes: int, names: List[str], distributions_priori: Optional[Tensor] = None
	):
		if distributions_priori is None:
			distributions_priori = ModelDistributions.uniform_distribution_onehot(history_size, nb_classes)

		self.names = names
		self.distributions_priori = distributions_priori
		self.data = {}

		self.reset()

	def reset(self):
		self.data = {
			name: [self.distributions_priori.clone(), 0] for name in self.names
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

	@staticmethod
	def uniform_distribution_onehot(history_size: int, nb_classes: int) -> Tensor:
		return torch.ones(history_size, nb_classes).cuda() / nb_classes

	@staticmethod
	def uniform_distribution_multihot(history_size: int, nb_classes: int) -> Tensor:
		return torch.ones(history_size, nb_classes).cuda() * 0.5
