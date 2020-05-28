import torch

from torch import Tensor
from torch.nn import Module
from typing import Callable

from ..mixup.mixup import MixUpMixer
from ..util.utils_match import same_shuffle, sharpen, merge_first_dimension


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
