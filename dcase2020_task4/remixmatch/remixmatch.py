import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import binary_cross_entropy_with_logits
from typing import Callable

from .ModelDistributions import ModelDistributions
from ..mixup.mixup import mixup_fn, MixUpMixer
from ..util.utils_match import normalize, same_shuffle, sharpen, merge_first_dimension, cross_entropy_with_logits


def remixmatch_fn(
	model: Module,
	batch_labeled: Tensor,
	labels: Tensor,
	batch_unlabeled: Tensor,
	strong_augm_fn_x: Callable,
	weak_augm_fn_x: Callable,
	distributions_coefs: Tensor,
	nb_augms_strong: int,
	sharpen_temp: float,
	mixup_alpha: float
) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor):
	"""
		ReMixMatch function.

		@params
			model: Current model used for compute guessed labels.
			batch_labeled: Batch from supervised dataset.
			labels: Labels of batch_labeled.
			batch_unlabeled: Unlabeled batch.
			strong_augm_fn_x: Strong augmentation function. Take a batch as input and return an strongly augmented version.
				This function should be a stochastic transformation.
			weak_augm_fn_x: Weak augmentation function. Take a batch as input and return an weakly augmented version.
				This function should be a stochastic transformation.
			nb_augms: Hyperparameter "K" used for compute guessed labels.
			sharpen_temp: Hyperparameter "T" used for temperature sharpening on guessed labels.
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
			guessed_labels[b] = sharpen(guessed_labels[b], sharpen_temp)

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
	loss_x = cross_entropy_with_logits(logits_x, targets_x)
	loss_u = cross_entropy_with_logits(logits_u, targets_u)
	loss_u1 = cross_entropy_with_logits(logits_u1, targets_u1)
	loss_r = cross_entropy_with_logits(logits_rot, targets_rot)

	loss = loss_x + lambda_u * loss_u + lambda_u1 * loss_u1 + lambda_r * loss_r

	return loss


class ReMixMatchLoss(Callable):
	def __init__(self, lambda_u: float = 1.5, lambda_u1: float = 0.5, lambda_r: float = 0.5, mode: str = "onehot"):
		self.lambda_u = lambda_u
		self.lambda_u1 = lambda_u1
		self.lambda_r = lambda_r
		self.mode = mode

		if self.mode == "onehot":
			self.criterion = cross_entropy_with_logits
		elif self.mode == "multihot":
			self.criterion = binary_cross_entropy_with_logits
		else:
			raise RuntimeError("Invalid argument \"mode = %s\". Use %s." % (mode, " or ".join(("onehot", "multihot"))))

	def __call__(
		self,
		logits_x: Tensor, targets_x: Tensor,
		logits_u: Tensor, targets_u: Tensor,
		logits_u1: Tensor, targets_u1: Tensor,
		logits_r: Tensor, targets_r: Tensor,
	) -> Tensor:
		loss_x = self.criterion(logits_x, targets_x)
		loss_u = self.criterion(logits_u, targets_u)
		loss_u1 = self.criterion(logits_u1, targets_u1)
		loss_r = self.criterion(logits_r, targets_r)

		loss = loss_x + self.lambda_u * loss_u + self.lambda_u1 * loss_u1 + self.lambda_r * loss_r

		return loss


class ReMixMatchMixer:
	def __init__(
		self,
		model: Module,
		weak_augm_fn: Callable,
		strong_augm_fn: Callable,
		nb_classes: int,
		nb_augms_strong: int,
		sharpen_temp: float,
		mixup_alpha: float,
		mode: str = "onehot",
	):
		self.model = model
		self.weak_augm_fn = weak_augm_fn
		self.strong_augm_fn = strong_augm_fn
		self.nb_augms_strong = nb_augms_strong
		self.sharpen_temp = sharpen_temp
		self.mode = mode

		self.distributions = ModelDistributions(history_size=128, nb_classes=nb_classes)
		self.mixup_mixer = MixUpMixer(alpha=mixup_alpha, apply_max=True)

		# NOTE: acti_fn must have the dim parameter !
		if self.mode == "onehot":
			self.acti_fn = torch.softmax
		elif self.mode == "multihot":
			self.acti_fn = lambda x, dim: x.sigmoid()
		else:
			raise RuntimeError("Invalid argument \"mode = %s\". Use %s." % (mode, " or ".join(("onehot", "multihot"))))

	def __call__(self, batch_labeled: Tensor, labels: Tensor, batch_unlabeled: Tensor) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor):
		return self.mix(batch_labeled, labels, batch_unlabeled)

	def mix(self, batch_s: Tensor, labels: Tensor, batch_u: Tensor) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor):
		with torch.no_grad():
			if batch_s.size() != batch_u.size():
				raise RuntimeError("Labeled and unlabeled batch must have the same size. (sizes: %s != %s)" % (
					str(batch_s.size()), str(batch_u.size())
				))

			# Apply augmentations
			x_augm = self.strong_augm_fn(batch_s)
			u_augm_s = torch.stack([self.strong_augm_fn(batch_u) for _ in range(self.nb_augms_strong)]).cuda()
			u_augm_w = self.weak_augm_fn(batch_u)

			# Compute guessed label
			logits = self.model(u_augm_w)
			guessed_labels = self.acti_fn(logits, dim=1)
			guessed_labels = guessed_labels * self.distributions.get_mean_pred("labeled") / self.distributions.get_mean_pred("unlabeled")
			if self.mode == "onehot":
				guessed_labels = normalize(guessed_labels, dim=1)
				guessed_labels = sharpen(guessed_labels, self.sharpen_temp, dim=1)
			guessed_labels_repeated = guessed_labels.repeat_interleave(self.nb_augms_strong, dim=0)

			# Get strongly augmented batch "u1"
			u1 = u_augm_s[0, :].clone()
			u1_labels = guessed_labels.clone()

			# Reshape u_augm_s of size (nb_augms, batch_size, sample_size, ...) to (nb_augms * batch_size, sample_size, ...)
			u_augm_s = merge_first_dimension(u_augm_s)

			# Concatenate strongly and weakly augmented data
			u_augm = torch.cat((u_augm_s, u_augm_w))
			guessed_labels_repeated = torch.cat((guessed_labels_repeated, guessed_labels))

			w = torch.cat((x_augm, u_augm), dim=0)
			w_labels = torch.cat((labels, guessed_labels_repeated), dim=0)

			# Shuffle batch and labels
			w, w_labels = same_shuffle([w, w_labels])

			x_len = len(x_augm)
			x_mixed, x_mixed_labels = self.mixup_mixer(x_augm, labels, w[:x_len], w_labels[:x_len])
			u_mixed, u_mixed_labels = self.mixup_mixer(u_augm, guessed_labels_repeated, w[x_len:], w_labels[x_len:])

			return x_mixed, x_mixed_labels, u_mixed, u_mixed_labels, u1, u1_labels
