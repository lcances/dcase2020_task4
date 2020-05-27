import torch

from torch import Tensor
from torch.nn import Module
from torch.nn.functional import binary_cross_entropy_with_logits
from typing import Callable

from ..mixup.mixup import mixup_fn, MixUpMixer
from ..util.utils_match import same_shuffle, sharpen, merge_first_dimension, cross_entropy_with_logits


def mixmatch_fn(
	model: Module,
	batch_labeled: Tensor,
	labels: Tensor,
	batch_unlabeled: Tensor,
	augment_fn_x: Callable,
	nb_augms: int,
	sharpen_temp: float,
	mixup_alpha: float
) -> (Tensor, Tensor, Tensor, Tensor):
	"""
		Apply Mixmatch function.

		@params
			model: Current model used for compute guessed labels.
			batch_labeled: Batch from supervised dataset.
			labels: Labels of batch_labeled.
			batch_unlabeled: Unlabeled batch.
			augment_fn: Augmentation function. Take a batch as input and return an augmented version.
				This function should be a stochastic transformation.
			nb_augms: Hyperparameter "K" used for compute guessed labels.
			sharpen_temp: Hyperparameter "T" used for temperature sharpening on guessed labels.
			mixup_alpha: Hyperparameter "alpha" used for MixUp.

		@returns
			A tuple (Batch mixed X, Labels mixed of X, Batch mixed U, Labels mixed of U).
	"""

	if batch_labeled.size() != batch_unlabeled.size():
		raise RuntimeError("Labeled and unlabeled batch must have the same size. (sizes: %s != %s)" % (
			str(batch_labeled.size()), str(batch_unlabeled.size())
		))

	with torch.no_grad():
		x_augm = torch.zeros(batch_labeled.size()).cuda()
		unlabeled_augm_size = [nb_augms] + list(batch_unlabeled.size())
		u_augm = torch.zeros(unlabeled_augm_size).cuda()
		guessed_labels = torch.zeros(labels.size()).cuda()

		for b, (x_sample, u_sample) in enumerate(zip(batch_labeled, batch_unlabeled)):
			x_augm[b] = augment_fn_x(x_sample)

			# TODO : replace by torch.stack for perf ?
			for k in range(nb_augms):
				u_augm[k, b] = augment_fn_x(u_sample)

			logits = model(u_augm[:, b])
			predictions = torch.softmax(logits, dim=1)
			guessed_labels[b] = torch.mean(predictions, dim=0)
			guessed_labels[b] = sharpen(guessed_labels[b], sharpen_temp)

		# Reshape u_augm of size (nb_augms, batch_size, sample_size, ...) to (nb_augms * batch_size, sample_size, ...)
		u_augm = merge_first_dimension(u_augm)

		# Duplicate guessed labels by "nb_augms"
		guessed_labels_repeated = guessed_labels.repeat_interleave(nb_augms, dim=0)

		w = torch.cat((x_augm, u_augm))
		w_labels = torch.cat((labels, guessed_labels_repeated))

		# Shuffle batch and labels
		w, w_labels = same_shuffle([w, w_labels])

		x_len = len(x_augm)
		x_mixed, x_mixed_labels = mixup_fn(x_augm, labels, w[:x_len], w_labels[:x_len], mixup_alpha)
		u_mixed, u_mixed_labels = mixup_fn(u_augm, guessed_labels_repeated, w[x_len:], w_labels[x_len:], mixup_alpha)

		return x_mixed, x_mixed_labels, u_mixed, u_mixed_labels


def mixmatch_loss(logits_x: Tensor, targets_x: Tensor, logits_u: Tensor, targets_u: Tensor, lambda_u: float):
	"""
		MixMatch loss.

		@params
			logits_x: Prediction of x_mixed.
			targets_x: Labels mixed.
			logits_u: Prediction of u_mixed.
			targets_u: Labels mixed.
			lambda_u: Hyperparameter to multiply with unsupervised loss component.

		@returns
			The MixMatch loss computed, a scalar Tensor.
	"""
	loss_x = cross_entropy_with_logits(logits_x, targets_x)

	pred_u = torch.softmax(logits_u, dim=1)
	loss_u = torch.mean((pred_u - targets_u) ** 2)

	return loss_x + lambda_u * loss_u


class MixMatchMixer(Callable):
	"""
		MixMatch class.
		Store hyperparameters and apply mixmatch_fn with call() or mix().
	"""
	def __init__(
		self,
		model: Module,
		augm_fn: Callable,
		nb_augms: int = 2,
		sharpen_temp: float = 0.5,
		mixup_alpha: float = 0.75,
		mode: str = "onehot",
	):
		self.model = model
		self.augm_fn = augm_fn
		self.nb_augms = nb_augms
		self.sharpen_temp = sharpen_temp
		self.mixup_mixer = MixUpMixer(alpha=mixup_alpha, apply_max=True)
		self.mode = mode

		# NOTE: acti_fn must have the dim parameter !
		if self.mode == "onehot":
			self.acti_fn = torch.softmax
		elif self.mode == "multihot":
			self.acti_fn = lambda x, dim: x.sigmoid()
		else:
			raise RuntimeError("Invalid argument \"mode = %s\". Use %s." % (mode, " or ".join(("onehot", "multihot"))))

	def __call__(self, batch_labeled: Tensor, labels: Tensor, batch_unlabeled: Tensor) -> (Tensor, Tensor, Tensor, Tensor):
		return self.mix(batch_labeled, labels, batch_unlabeled)

	def mix(self, batch_s: Tensor, labels: Tensor, batch_u: Tensor) -> (Tensor, Tensor, Tensor, Tensor):
		with torch.no_grad():
			if batch_s.size() != batch_u.size():
				raise RuntimeError("Labeled and unlabeled batch must have the same size. (sizes: %s != %s)" % (
					str(batch_s.size()), str(batch_u.size())
				))

			# Apply augmentations
			x_augm = self.augm_fn(batch_s)
			u_augm = torch.stack([self.augm_fn(batch_u) for _ in range(self.nb_augms)]).cuda()

			# Compute guessed label
			logits = torch.stack([self.model(u_augm[k]) for k in range(self.nb_augms)]).cuda()
			predictions = self.acti_fn(logits, dim=2)
			guessed_labels = predictions.mean(dim=0)
			if self.mode == "onehot":
				guessed_labels = sharpen(guessed_labels, self.sharpen_temp, dim=1)
			guessed_labels_repeated = guessed_labels.repeat_interleave(self.nb_augms, dim=0)

			# Reshape "u_augm" of size (nb_augms, batch_size, sample_size, ...) to (nb_augms * batch_size, sample_size, ...)
			u_augm = merge_first_dimension(u_augm)

			w = torch.cat((x_augm, u_augm))
			w_labels = torch.cat((labels, guessed_labels_repeated))

			# Shuffle batch and labels
			w, w_labels = same_shuffle([w, w_labels])

			x_len = len(x_augm)
			x_mixed, x_mixed_labels = self.mixup_mixer(x_augm, labels, w[:x_len], w_labels[:x_len])
			u_mixed, u_mixed_labels = self.mixup_mixer(u_augm, guessed_labels_repeated, w[x_len:], w_labels[x_len:])

			return x_mixed, x_mixed_labels, u_mixed, u_mixed_labels


class MixMatchLoss(Callable):
	def __init__(self, lambda_u: float = 1.0, mode: str = "onehot", criterion_unsupervised: str = "l2norm"):
		self.lambda_u = lambda_u
		self.mode = mode
		self.unsupervised_loss_mode = criterion_unsupervised

		if self.mode == "onehot":
			self.acti_fn = lambda x: torch.softmax(x, dim=1)
			self.criterion_s = cross_entropy_with_logits
			if criterion_unsupervised == "l2norm":
				self.criterion_u = lambda logits_u, targets_u: torch.mean((self.acti_fn(logits_u) - targets_u) ** 2)
			elif criterion_unsupervised == "crossentropy":
				self.criterion_u = cross_entropy_with_logits
			else:
				raise RuntimeError("Invalid argument \"mode = %s\". Use %s." % (mode, " or ".join(("onehot", "multihot"))))

		elif self.mode == "multihot":
			self.acti_fn = torch.sigmoid
			self.criterion_s = binary_cross_entropy_with_logits
			self.criterion_u = binary_cross_entropy_with_logits

		else:
			raise RuntimeError("Invalid argument \"mode = %s\". Use %s." % (mode, " or ".join(("onehot", "multihot"))))

	def __call__(self, logits_x: Tensor, targets_x: Tensor, logits_u: Tensor, targets_u: Tensor) -> Tensor:
		loss_x = self.criterion_s(logits_x, targets_x)
		loss_x = loss_x.mean()

		loss_u = self.criterion_u(logits_u, targets_u)
		loss_u = loss_u.mean()

		return loss_x + self.lambda_u * loss_u
