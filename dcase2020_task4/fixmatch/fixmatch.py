import torch

from torch import Tensor
from torch.nn import CrossEntropyLoss

from ..util.utils_match import to_class_num


# TODO : rem
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
