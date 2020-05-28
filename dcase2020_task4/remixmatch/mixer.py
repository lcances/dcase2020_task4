import torch
from torch import Tensor
from torch.nn import Module
from typing import Callable

from .ModelDistributions import ModelDistributions
from ..mixup.mixup import MixUpMixer
from ..util.utils_match import normalize, same_shuffle, sharpen, merge_first_dimension


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

		self.distributions = ModelDistributions(history_size=128, nb_classes=nb_classes, names=["labeled", "unlabeled"])
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
