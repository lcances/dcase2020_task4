
import torch

from abc import ABC
from argparse import Namespace
from torch import Tensor
from typing import List, Union

from dcase2020_task4.util.utils_match import normalized


class DistributionAlignment(ABC):
	"""
		Abstract for compute mean output distributions of a model and use it to align other predictions.
	"""

	def __init__(self, distributions_init: Tensor, names: List[str]):
		self.names = names
		self.distributions_init = distributions_init
		self.data = {}

		self.reset()

	def __call__(self, batch: Tensor, dim: Union[int, tuple]) -> Tensor:
		return self.apply_distribution_alignment(batch, dim)

	def apply_distribution_alignment(self, batch: Tensor, dim: Union[int, tuple]) -> Tensor:
		raise NotImplementedError("Abstract method")

	def reset(self):
		self.data = {
			name: [self.distributions_init.clone(), 0] for name in self.names
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


class DistributionAlignmentOnehot(DistributionAlignment):
	def __init__(self, history_size: int, shape: List[int], names: List[str]):
		distributions_init = DistributionAlignmentOnehot.uniform_distribution_onehot(history_size, shape)
		super().__init__(distributions_init, names)

	@staticmethod
	def from_args(args: Namespace) -> 'DistributionAlignmentOnehot':
		return DistributionAlignmentOnehot(
			history_size=args.history_size,
			shape=[args.nb_classes],
			names=["labeled", "unlabeled"],
		)

	def apply_distribution_alignment(self, batch: Tensor, dim: Union[int, tuple]) -> Tensor:
		batch = batch.clone()
		coefficients = self.get_avg_pred("labeled") / self.get_avg_pred("unlabeled")

		batch = batch * coefficients
		batch = normalized(batch, dim)

		return batch

	@staticmethod
	def uniform_distribution_onehot(history_size: int, shape: List[int]) -> Tensor:
		return torch.ones([history_size] + shape).cuda() / shape[-1]


class DistributionAlignmentMultihot(DistributionAlignment):
	def __init__(self, history_size: int, shape: List[int], names: List[str]):
		distributions_init = DistributionAlignmentMultihot.uniform_distribution_multihot(history_size, shape)
		super().__init__(distributions_init, names)

	@staticmethod
	def from_args(args: Namespace) -> 'DistributionAlignmentMultihot':
		return DistributionAlignmentMultihot(
			history_size=args.history_size,
			shape=[args.nb_classes],
			names=["labeled", "unlabeled"],
		)

	def apply_distribution_alignment(self, batch: Tensor, dim: Union[int, tuple]) -> Tensor:
		batch = batch.clone()
		coefficients = self.get_avg_pred("labeled") / self.get_avg_pred("unlabeled")

		prev_norm = batch.norm(p=1, dim=dim, keepdim=True)
		# Apply coefficients
		batch = batch * coefficients
		# Normalize
		batch = normalized(batch, dim)
		# Increase probability with old norm
		batch = batch * prev_norm
		# If a distribution contains a value above 1.0, it need to be rescale
		batch = batch / batch.max(dim=dim, keepdim=True)[0].clamp(min=1.0)

		return batch

	@staticmethod
	def uniform_distribution_multihot(history_size: int, shape: List[int]) -> Tensor:
		return torch.ones([history_size] + shape).cuda() * 0.5
