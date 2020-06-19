import numpy as np
import torch

from torch import Tensor


class MixUpMixer:
	def __init__(self, alpha: float = 0.75, apply_max: bool = True):
		self.alpha = alpha
		self.apply_max = apply_max

	def __call__(self, batch_1: Tensor, labels_1: Tensor, batch_2: Tensor, labels_2: Tensor) -> (Tensor, Tensor):
		return self.mix(batch_1, labels_1, batch_2, labels_2)

	def mix(self, batch_1: Tensor, labels_1: Tensor, batch_2: Tensor, labels_2: Tensor) -> (Tensor, Tensor):
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
			if batch_1.size() != batch_2.size() or labels_1.size() != labels_2.size():
				raise RuntimeError("Batches and labels must have the same size for MixUp.")

			lambda_ = np.random.beta(self.alpha, self.alpha)
			if self.apply_max:
				lambda_ = max(lambda_, 1.0 - lambda_)
			batch_mixed = batch_1 * lambda_ + batch_2 * (1.0 - lambda_)
			labels_mixed = labels_1 * lambda_ + labels_2 * (1.0 - lambda_)

			return batch_mixed, labels_mixed


MixUpMixerMonoLabel = MixUpMixer
