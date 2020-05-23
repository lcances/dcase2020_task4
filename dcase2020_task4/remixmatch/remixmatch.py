import torch
from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from typing import Callable

from .ModelDistributions import ModelDistributions
from ..mixup.mixup import mixup_fn
from ..util.match_utils import normalize, same_shuffle, sharpen, merge_first_dimension, cross_entropy_with_one_hot


def remixmatch_fn(
	model: Module,
	batch_labeled: Tensor,
	labels: Tensor,
	batch_unlabeled: Tensor,
	strong_augm_fn_x: Callable,
	weak_augm_fn_x: Callable,
	distributions_coefs: Tensor,
	nb_augms_strong: int,
	sharpen_val: float,
	mixup_alpha: float
) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor):
	"""
		ReMixMatch function.

		@params
			model: Current model used for compute guessed labels.
			batch_labeled: Batch from supervised dataset.
			labels: Labels of batch_labeled.
			batch_unlabeled: Unlabeled batch.
			strong_augm_fn_x: Strong augmentation function. Take a sample as input and return an strongly augmented version.
				This function should be a stochastic transformation.
			weak_augm_fn_x: Weak augmentation function. Take a sample as input and return an weakly augmented version.
				This function should be a stochastic transformation.
			nb_augms: Hyperparameter "K" used for compute guessed labels.
			sharpen_val: Hyperparameter "T" used for temperature sharpening on guessed labels.
			mixup_alpha: Hyperparameter "alpha" used for MixUp.

		@returns
			A tuple (Batch mixed X, Labels mixed of X, Batch mixed U, Labels mixed of U, Batch strongly augmented U1, Labels of U1).
	"""
	if batch_labeled.size() != batch_unlabeled.size():
		raise RuntimeError("Labeled and unlabeled batch must have the same size. (sizes: %s != %s)" % (
			str(batch_labeled.size()), str(batch_unlabeled.size())
		))

	with torch.no_grad():
		unlabeled_augm_s_size = [nb_augms_strong] + list(batch_unlabeled.size())

		x_augm = torch.zeros(batch_labeled.size()).cuda()
		u_augm_s = torch.zeros(unlabeled_augm_s_size).cuda()
		u_augm_w = torch.zeros(batch_unlabeled.size()).cuda()

		for b, (x_sample, u_sample) in enumerate(zip(batch_labeled, batch_unlabeled)):
			x_augm[b] = strong_augm_fn_x(x_sample)
			for k in range(nb_augms_strong):
				u_augm_s[k, b] = strong_augm_fn_x(u_sample)
			u_augm_w[b] = weak_augm_fn_x(u_sample)

		# Guess labels
		logits = model(u_augm_w)
		guessed_labels = torch.softmax(logits, dim=1)

		for b in range(len(guessed_labels)):
			# Distribution alignment
			guessed_labels[b] = normalize(guessed_labels[b] * distributions_coefs)
			# Sharpening
			guessed_labels[b] = sharpen(guessed_labels[b], sharpen_val)

		# Get strongly augmented batch "U1"
		u1 = u_augm_s[0, :].clone()
		u1_labels = guessed_labels

		# Reshape u_augm_s of size (nb_augms, batch_size, sample_size, ...) to (nb_augms * batch_size, sample_size, ...)
		u_augm_s = merge_first_dimension(u_augm_s)

		# Duplicate labels for u_augm_s
		guessed_labels_repeated = guessed_labels.repeat_interleave(nb_augms_strong, dim=0)

		# Concatenate strongly and weakly augmented data
		u_augm = torch.cat((u_augm_s, u_augm_w))
		guessed_labels_repeated = torch.cat((guessed_labels_repeated, guessed_labels))

		w = torch.cat((x_augm, u_augm), dim=0)
		w_labels = torch.cat((labels, guessed_labels_repeated), dim=0)

		# Shuffle batch and labels
		w, w_labels = same_shuffle([w, w_labels])

		x_mixed, x_mixed_labels = mixup_fn(
			x_augm, labels, w[:len(x_augm)], w_labels[:len(x_augm)], mixup_alpha
		)
		u_mixed, u_mixed_labels = mixup_fn(
			u_augm, guessed_labels_repeated, w[len(x_augm):], w_labels[len(x_augm):], mixup_alpha
		)

		return x_mixed, x_mixed_labels, u_mixed, u_mixed_labels, u1, u1_labels


def remixmatch_loss(
	logits_x: Tensor,
	targets_x: Tensor,
	logits_u: Tensor,
	targets_u: Tensor,
	logits_u1: Tensor,
	targets_u1: Tensor,
	logits_rot: Tensor,
	targets_rot: Tensor,
	lambda_u: float,
	lambda_u1: float,
	lambda_r: float,
) -> Tensor:
	"""
		ReMixMatch loss.

		@params
			logits_x: Prediction of x_mixed.
			targets_x: Labels mixed.
			logits_u: Prediction of u_mixed.
			targets_u: Labels mixed.
			logits_u1: Prediction of u1_labels.
			targets_u1: Pseudo-labels of strong augmented unsupervised batch.
			logits_rot: Prediction of rotation augmentation.
			targets_rot: Rotation labels.
			lambda_u: Hyperparameter to multiply with unsupervised loss component.
			lambda_u1: Hyperparameter to multiply with strong augmented unsupervised loss component.
			lambda_r: Hyperparameter to multiply with rotation loss component.

		@returns
			The ReMixMatch loss computed, a scalar Tensor.
	"""
	loss_x = cross_entropy_with_one_hot(logits_x, targets_x)
	loss_u = cross_entropy_with_one_hot(logits_u, targets_u)
	loss_u1 = cross_entropy_with_one_hot(logits_u1, targets_u1)
	loss_r = cross_entropy_with_one_hot(logits_rot, targets_rot)

	loss = loss_x + lambda_u * loss_u + lambda_u1 * loss_u1 + lambda_r * loss_r

	return loss


class ReMixMatch:
	def __init__(
		self,
		model: Module,
		optim: Optimizer,
		weak_augm_fn_x: Callable,
		strong_augm_fn_x: Callable,
		nb_classes: int,
		nb_augms_strong: int,
		sharpen_val: float,
		mixup_alpha: float
	):
		self.model = model
		self.optim = optim
		self.weak_augm_fn_x = weak_augm_fn_x
		self.strong_augm_fn_x = strong_augm_fn_x
		self.nb_augms_strong = nb_augms_strong
		self.sharpen_val = sharpen_val
		self.mixup_alpha = mixup_alpha
		self.distributions = ModelDistributions(nb_classes)

	def __call__(self, batch_labeled: Tensor, labels: Tensor, batch_unlabeled: Tensor) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor):
		return self.mix(batch_labeled, labels, batch_unlabeled)

	def mix(self, batch_labeled: Tensor, labels: Tensor, batch_unlabeled: Tensor) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor):
		distributions_coefficients = self.distributions.get_mean_pred("labeled") / self.distributions.get_mean_pred("unlabeled")
		return remixmatch_fn(
			self.model,
			batch_labeled,
			labels,
			batch_unlabeled,
			self.strong_augm_fn_x,
			self.weak_augm_fn_x,
			distributions_coefficients,
			self.nb_augms_strong,
			self.sharpen_val,
			self.mixup_alpha,
		)
