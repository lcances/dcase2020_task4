import torch

from torch import nn, Tensor
from torch.nn import BCELoss
from typing	import Callable


class SupervisedLossLoc(Callable):
	def __init__(self, lambda_u: float = 1.0, reduce: str = "mean"):
		assert reduce in ["mean", "sum"], "support only \"mean\" and \"sum\""

		self.lambda_u = lambda_u
		if reduce == "mean":
			self.reduce_fn = torch.mean
		else:
			self.reduce_fn = torch.sum

		self.criterion_weak = BCELoss(reduction="none")
		self.criterion_strong = BCELoss(reduction="none")

	def __call__(
		self, pred_weak: Tensor, label_weak: Tensor, pred_strong: Tensor, label_strong: Tensor
	) -> (Tensor, Tensor, Tensor):
		# Weak label loss
		loss_weak = self.criterion_weak(pred_weak, label_weak)
		loss_weak = self.reduce_fn(loss_weak, dim=1)

		# Strong label loss
		strong_mask = self.get_strong_mask(label_strong)
		loss_strong = self.criterion_strong(pred_strong, label_strong)
		loss_strong = self.reduce_fn(loss_strong, dim=(1, 2))

		# Final loss
		loss_weak = self.reduce_fn(loss_weak)
		loss_strong = self.reduce_fn(strong_mask * loss_strong)
		loss = loss_weak + self.lambda_u * loss_strong

		return loss, loss_weak, loss_strong

	def get_strong_mask(self, label_strong: Tensor) -> Tensor:
		return label_strong.sum(dim=(1, 2)).clamp(0, 1)


def weak_synth_loss(
	logits_weak: Tensor, logits_strong: Tensor, y_weak: Tensor, y_strong: Tensor, reduce: str = "mean"
) -> (Tensor, Tensor, Tensor):
	assert reduce in ["mean", "sum"], "support only \"mean\" and \"sum\""

	#  Reduction function
	if reduce == "mean":
		reduce_fn = torch.mean
	else:
		reduce_fn = torch.sum

	# based on Binary Cross Entropy loss
	weak_criterion = nn.BCEWithLogitsLoss(reduction="none")
	strong_criterion = nn.BCEWithLogitsLoss(reduction="none")

	# calc separate loss function
	weak_bce = weak_criterion(logits_weak, y_weak)
	strong_bce = strong_criterion(logits_strong, y_strong)

	weak_bce = reduce_fn(weak_bce, dim=1)
	strong_bce = reduce_fn(strong_bce, dim=(1, 2))

	# calc strong mask
	strong_mask = torch.clamp(torch.sum(y_strong, dim=(1, 2)), 0, 1)  # vector of 0 or 1
	#     strong_mask = strong_mask.detach() # declared not to need gradients

	# Output the different loss for logging purpose
	weak_loss = reduce_fn(weak_bce)
	strong_loss = reduce_fn(strong_mask * strong_bce)
	total_loss = reduce_fn(weak_bce + strong_mask * strong_bce)

	return weak_loss, strong_loss, total_loss
