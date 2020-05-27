import torch

from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.nn.functional import binary_cross_entropy_with_logits, one_hot
from typing import Callable

from ..util.utils_match import to_class_num, cross_entropy_with_logits


def fixmatch_loss(
	logits_s_weak: Tensor,
	labels: Tensor,
	logits_u_weak: Tensor,
	logits_u_strong: Tensor,
	threshold: float,
	lambda_u: float
) -> Tensor:
	"""
		FixMatch loss.

		@params
			logits_labeled_weak: Output of the model for weakly augmented labeled batch.
			labels: True labels of the current labeled batch.
			logits_unlabeled_weak: Output of the model for weakly augmented unlabeled batch.
			logits_unlabeled_strong: Output of the model for strongly augmented unlabeled batch.
			threshold: Hyperparameter for a compute loss component if the maximum confidence is above threshold.
			lambda_u: Hyperparameter to multiply with unsupervised loss component.

		@returns
			The FixMatch loss computed, a scalar Tensor.
	"""
	if logits_u_weak.size() != logits_u_strong.size():
		raise RuntimeError("Weak predictions and strong predictions must have the same size.")

	ce = CrossEntropyLoss()

	# Supervised loss
	labels_nums = to_class_num(labels)
	loss_s = torch.mean(ce(logits_s_weak, labels_nums))

	# Unsupervised loss
	pred_unlabeled_weak = torch.softmax(logits_u_weak, dim=1)
	max_values, guessed_labels_nums = pred_unlabeled_weak.max(dim=1)
	mask = (max_values > threshold).float()
	loss_u = ce(logits_u_strong, guessed_labels_nums) * mask
	loss_u = torch.mean(loss_u)

	loss = loss_s + lambda_u * loss_u
	return loss


class FixMatchLoss(Callable):
	def __init__(
		self,
		lambda_u: float = 1.0,
		threshold_mask: float = 0.95,
		threshold_multihot: float = 0.9,
		mode: str = "onehot",
	):
		self.lambda_u = lambda_u
		self.threshold_mask = threshold_mask
		self.threshold_multihot = threshold_multihot
		self.mode = mode

		if self.mode == "onehot":
			self.acti_fn = lambda x: torch.softmax(x, dim=1)
			self.criterion_s = cross_entropy_with_logits
			self.criterion_u = cross_entropy_with_logits
		elif self.mode == "multihot":
			self.acti_fn = torch.sigmoid
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
