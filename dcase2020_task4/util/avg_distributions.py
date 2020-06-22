import torch
from torch import Tensor
from typing import List


class AvgDistributions:
	"""
		Compute mean output distributions of a model.
	"""

	def __init__(
		self, history_size: int, shape: List[int], mode: str, names: List[str]
	):
		if mode == "onehot":
			distributions_priori = AvgDistributions.uniform_distribution_onehot(history_size, shape)
		elif mode == "multihot":
			distributions_priori = AvgDistributions.uniform_distribution_multihot(history_size, shape)
		else:
			raise RuntimeError("Invalid argument \"mode = %s\". Use %s." % (mode, " or ".join(("onehot", "multihot"))))

		self.names = names
		self.distributions_priori = distributions_priori
		self.data = {}

		self.reset()

	@staticmethod
	def from_edict(hparams) -> 'AvgDistributions':
		return AvgDistributions(
			history_size=hparams.history_size,
			shape=[hparams.nb_classes],
			mode=hparams.mode,
			names=["labeled", "unlabeled"],
		)

	def reset(self):
		self.data = {
			name: [self.distributions_priori.clone(), 0] for name in self.names
		}

	def add_batch_pred(self, batch: Tensor, name: str):
		with torch.no_grad():
			for pred in batch:
				self.add_pred(pred, name)

	def add_pred(self, pred: Tensor, name: str):
		with torch.no_grad():
			distributions, index = self.data[name]
			distributions[index] = pred.clone()
			index = (index + 1) % len(distributions)
			self.data[name][1] = index

	def get_avg_pred(self, name: str) -> Tensor:
		distributions, _ = self.data[name]
		return torch.mean(distributions, dim=0)

	@staticmethod
	def uniform_distribution_onehot(history_size: int, shape: List[int]) -> Tensor:
		return torch.ones([history_size] + shape).cuda() / shape[-1]

	@staticmethod
	def uniform_distribution_multihot(history_size: int, shape: List[int]) -> Tensor:
		return torch.ones([history_size] + shape).cuda() * 0.5
