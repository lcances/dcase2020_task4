import numpy as np
from torch import Tensor


def mixup_fn(
	batch_1: Tensor,
	labels_1: Tensor,
	batch_2: Tensor,
	labels_2: Tensor,
	alpha: float,
	apply_max: bool = True
) -> (Tensor, Tensor):
	"""
		MixUp function.

		@params
			batch_1: First batch.
			labels_1: Labels of batch_1.
			batch_2: Second batch.
			labels_2: Labels of batch_2.
			alpha: Hyperparameter of Beta distribution for sample lambda coefficient.
			apply_max: If True, the greatest coefficient will be applied on first batch.

		@returns
			A tuple (batch mixed, labels mixed).
	"""
	if batch_1.size() != batch_2.size() or labels_1.size() != labels_2.size():
		raise RuntimeError("Batches and labels must have the same size for MixUp.")

	lambda_ = np.random.beta(alpha, alpha)
	if apply_max:
		lambda_ = max(lambda_, 1.0 - lambda_)
	batch_mixed = batch_1 * lambda_ + batch_2 * (1.0 - lambda_)
	labels_mixed = labels_1 * lambda_ + labels_2 * (1.0 - lambda_)

	return batch_mixed, labels_mixed
