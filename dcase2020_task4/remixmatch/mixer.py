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
		acti_fn: Callable,
		weak_augm_fn: Callable,
		strong_augm_fn: Callable,
		nb_classes: int,
		nb_augms_strong: int,
		sharpen_temp: float,
		mixup_alpha: float,
		mode: str = "onehot",
	):
		self.model = model
		self.acti_fn = acti_fn
		self.weak_augm_fn = weak_augm_fn
		self.strong_augm_fn = strong_augm_fn
		self.nb_augms_strong = nb_augms_strong
		self.sharpen_temp = sharpen_temp
		self.mode = mode

		self.distributions = ModelDistributions(history_size=128, nb_classes=nb_classes, names=["labeled", "unlabeled"])
		self.mixup_mixer = MixUpMixer(alpha=mixup_alpha, apply_max=True)

	def __call__(self, batch_s: Tensor, labels_s: Tensor, batch_u: Tensor) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor):
		return self.mix(batch_s, labels_s, batch_u)

	def mix(self, batch_s: Tensor, labels_s: Tensor, batch_u: Tensor) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor):
		with torch.no_grad():
			if batch_s.size() != batch_u.size():
				raise RuntimeError("Labeled and unlabeled batch must have the same size. (sizes: %s != %s)" % (
					str(batch_s.size()), str(batch_u.size())
				))

			# Apply augmentations
			batch_s_strong = self.strong_augm_fn(batch_s)
			batch_u_strong = torch.stack([self.strong_augm_fn(batch_u) for _ in range(self.nb_augms_strong)]).cuda()
			batch_u_weak = self.weak_augm_fn(batch_u)

			# Compute guessed label
			logits_u_weak = self.model(batch_u_weak)
			labels_u_guessed = self.acti_fn(logits_u_weak, dim=1)
			labels_u_guessed = labels_u_guessed * self.distributions.get_mean_pred("labeled") / self.distributions.get_mean_pred("unlabeled")
			if self.mode == "onehot":
				labels_u_guessed = normalize(labels_u_guessed, dim=1)
				labels_u_guessed = sharpen(labels_u_guessed, self.sharpen_temp, dim=1)
			labels_u_guessed_repeated = labels_u_guessed.repeat_interleave(self.nb_augms_strong, dim=0)

			# Get strongly augmented batch "batch_u1"
			batch_u1 = batch_u_strong[0, :].clone()
			labels_u1 = labels_u_guessed.clone()

			# Reshape batch_u_strong of size (nb_augms, batch_size, sample_size...) to (nb_augms * batch_size, sample_size...)
			batch_u_strong = merge_first_dimension(batch_u_strong)

			# Concatenate strongly and weakly augmented data from batch_u
			batch_u_cat = torch.cat((batch_u_strong, batch_u_weak))
			labels_u_cat = torch.cat((labels_u_guessed_repeated, labels_u_guessed))

			batch_w = torch.cat((batch_s_strong, batch_u_cat), dim=0)
			labels_w = torch.cat((labels_s, labels_u_cat), dim=0)

			# Shuffle batch and labels
			batch_w, labels_w = same_shuffle([batch_w, labels_w])

			len_s = len(batch_s_strong)
			batch_s_mixed, labels_s_mixed = self.mixup_mixer(
				batch_s_strong, labels_s, batch_w[:len_s], labels_w[:len_s]
			)
			batch_u_mixed, labels_u_mixed = self.mixup_mixer(
				batch_u_cat, labels_u_cat, batch_w[len_s:], labels_w[len_s:]
			)

			return batch_s_mixed, labels_s_mixed, batch_u_mixed, labels_u_mixed, batch_u1, labels_u1
