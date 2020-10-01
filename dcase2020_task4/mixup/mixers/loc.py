import numpy as np
import torch

from argparse import Namespace
from torch import Tensor
from typing import Callable, List


class MixUpMixerLoc(Callable):
	def __init__(self, alpha: float = 0.75, apply_max: bool = True):
		self.alpha = alpha
		self.apply_max = apply_max

	@staticmethod
	def from_args(args: Namespace) -> 'MixUpMixerLoc':
		return MixUpMixerLoc(args.mixup_alpha, True)

	def __call__(
		self, batch_1: Tensor, labels_1: List[Tensor], batch_2: Tensor, labels_2: List[Tensor]
	) -> (Tensor, List[Tensor]):
		return self.mix(batch_1, labels_1, batch_2, labels_2)

	def mix(
		self, batch_1: Tensor, labels_1: List[Tensor], batch_2: Tensor, labels_2: List[Tensor]
	) -> (Tensor, List[Tensor]):
		"""
			MixUp method.

			@params
				batch_1: First batch.
				labels_1: Labels of batch_1.
				batch_2: Second batch.
				labels_2: Labels of batch_2.

			@returns
				A tuple (batch mixed, labels mixed).
		"""
		with torch.no_grad():
			if batch_1.size() != batch_2.size() or len(labels_1) != len(labels_2):
				raise RuntimeError("Batches and labels must have the same size for MixUp.")
			for label_1, label_2 in zip(labels_1, labels_2):
				if label_1.size() != label_2.size():
					raise RuntimeError("Labels must have the same size for MixUp.")

			lambda_ = np.random.beta(self.alpha, self.alpha)
			if self.apply_max:
				lambda_ = max(lambda_, 1.0 - lambda_)

			batch_mixed = batch_1 * lambda_ + batch_2 * (1.0 - lambda_)
			labels_mixed = [
				label_1 * lambda_ + label_2 * (1.0 - lambda_)
				for label_1, label_2 in zip(labels_1, labels_2)
			]

			return batch_mixed, labels_mixed
