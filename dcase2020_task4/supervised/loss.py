import torch

from torch import nn, Tensor


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
