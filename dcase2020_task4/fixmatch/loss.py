import torch

from torch import Tensor
from torch.nn.functional import binary_cross_entropy_with_logits, one_hot
from typing import Callable

from ..util.utils_match import cross_entropy_with_logits


class FixMatchLoss(Callable):
	def __init__(
		self,
		acti_fn: Callable,
		lambda_u: float = 1.0,
		threshold_mask: float = 0.95,
		threshold_multihot: float = 0.9,
		mode: str = "onehot",
	):
		self.acti_fn = acti_fn
		self.lambda_u = lambda_u
		self.threshold_mask = threshold_mask
		self.threshold_multihot = threshold_multihot
		self.mode = mode

		if self.mode == "onehot":
			self.criterion_s = cross_entropy_with_logits
			self.criterion_u = cross_entropy_with_logits
		elif self.mode == "multihot":
			self.criterion_s = binary_cross_entropy_with_logits
			self.criterion_u = binary_cross_entropy_with_logits
		else:
			raise RuntimeError("Invalid argument \"mode = %s\". Use %s." % (self.mode, " or ".join(("onehot", "multihot"))))

	def __call__(
		self,
		logits_s_weak: Tensor,
		labels: Tensor,
		logits_u_weak: Tensor,
		logits_u_strong: Tensor,
	) -> Tensor:
		if logits_s_weak.size() != labels.size():
			raise RuntimeError("Weak predictions and labels must have the same size.")
		if logits_u_weak.size() != logits_u_strong.size():
			raise RuntimeError("Weak predictions and strong predictions must have the same size.")

		# Supervised loss
		loss_s = self.criterion_s(logits_s_weak, labels)
		loss_s = loss_s.mean()

		# Unsupervised loss
		pred_u_weak = self.acti_fn(logits_u_weak)
		max_values, guessed_labels_nums = pred_u_weak.max(dim=1)

		if self.mode == "onehot":
			nb_classes = labels.size()[1]
			guessed_labels = one_hot(guessed_labels_nums, nb_classes)
		elif self.mode == "multihot":
			guessed_labels = (pred_u_weak > self.threshold_multihot).float()
		else:
			raise RuntimeError("Invalid argument \"mode = %s\". Use %s." % (self.mode, " or ".join(("onehot", "multihot"))))

		mask = (max_values > self.threshold_mask).float()
		loss_u = self.criterion_u(logits_u_strong, guessed_labels)
		loss_u *= mask
		loss_u = loss_u.mean()

		loss = loss_s + self.lambda_u * loss_u
		return loss
