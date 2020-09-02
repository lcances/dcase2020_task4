
import torch

from torch import Tensor
from typing import List, Union

from dcase2020_task4.util.utils_match import normalized


class DistributionAlignment:
	"""
		Compute mean output distributions of a model and use it to modify a prediction.
	"""

	def __init__(
		self, history_size: int, shape: List[int], mode: str, names: List[str]
	):
		if mode == "onehot":
			distributions_priori = DistributionAlignment.uniform_distribution_onehot(history_size, shape)
		elif mode == "multihot":
			distributions_priori = DistributionAlignment.uniform_distribution_multihot(history_size, shape)
		else:
			raise RuntimeError("Invalid argument \"mode = %s\". Use %s." % (mode, " or ".join(("onehot", "multihot"))))

		self.mode = mode
		self.names = names
		self.distributions_priori = distributions_priori
		self.data = {}

		self.reset()

	@staticmethod
	def from_edict(hparams) -> 'DistributionAlignment':
		return DistributionAlignment(
			history_size=hparams.history_size,
			shape=[hparams.nb_classes],
			mode=hparams.mode,
			names=["labeled", "unlabeled"],
		)

	def __call__(self, batch: Tensor, dim: Union[int, tuple]) -> Tensor:
		return self.apply_distribution_alignment(batch, dim)

	def apply_distribution_alignment(self, batch: Tensor, dim: Union[int, tuple]) -> Tensor:
		batch = batch.clone()
		coefficients = self.get_avg_pred("labeled") / self.get_avg_pred("unlabeled")

		if self.mode == "onehot":
			batch = batch * coefficients
			batch = normalized(batch, dim)
		elif self.mode == "multihot":
			prev_norm = batch.norm(p=1, dim=dim, keepdim=True)
			# Apply coefficients
			batch = batch * coefficients
			# Normalize
			batch = normalized(batch, dim)
			# Increase probability with old norm
			batch = batch * prev_norm
			# If a distribution contains a value above 1.0, it need to be rescale
			batch = batch / batch.max(dim=dim, keepdim=True)[0].clamp(min=1.0)
		else:
			raise RuntimeError("Invalid mode %s" % self.mode)

		return batch

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
