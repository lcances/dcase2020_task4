import torch

from torch import Tensor
from torch.nn import Module
from typing import Callable

from dcase2020_task4.mixup.mixer import MixUpMixer
from dcase2020_task4.util.utils_match import same_shuffle, sharpen, merge_first_dimension


class MixMatchMixer(Callable):
	"""
		MixMatch class.
		Store hyperparameters and apply mixmatch_fn with call() or mix().
	"""
	def __init__(
		self,
		model: Module,
		acti_fn: Callable[[Tensor, int], Tensor],
		augm_fn: Callable[[Tensor], Tensor],
		nb_augms: int = 2,
		sharpen_temp: float = 0.5,
		mixup_alpha: float = 0.75,
		mode: str = "onehot",
	):
		self.model = model
		self.acti_fn = acti_fn
		self.augm_fn = augm_fn
		self.nb_augms = nb_augms
		self.sharpen_temp = sharpen_temp
		self.mixup_mixer = MixUpMixer(alpha=mixup_alpha, apply_max=True)
		self.mode = mode

	def __call__(self, batch_s: Tensor, labels_s: Tensor, batch_u: Tensor) -> (Tensor, Tensor, Tensor, Tensor):
		return self.mix(batch_s, labels_s, batch_u)

	def mix(self, batch_s: Tensor, labels_s: Tensor, batch_u: Tensor) -> (Tensor, Tensor, Tensor, Tensor):
		with torch.no_grad():
			if batch_s.size() != batch_u.size():
				raise RuntimeError("Labeled and unlabeled batch must have the same size. (sizes: %s != %s)" % (
					str(batch_s.size()), str(batch_u.size())
				))

			# Apply augmentations
			batch_s_augm = self.augm_fn(batch_s)
			batch_u_augm = torch.stack([self.augm_fn(batch_u) for _ in range(self.nb_augms)]).cuda()

			# Compute guessed label
			logits_u_augm = torch.stack([self.model(batch_u_augm[k]) for k in range(self.nb_augms)]).cuda()
			predictions_u_augm = self.acti_fn(logits_u_augm, dim=2)
			labels_u_guessed = predictions_u_augm.mean(dim=0)
			if self.mode == "onehot":
				labels_u_guessed = sharpen(labels_u_guessed, self.sharpen_temp, dim=1)
			labels_u_guessed_repeated = labels_u_guessed.repeat_interleave(self.nb_augms, dim=0)

			# Reshape "batch_u_augm" of size (nb_augms, batch_size, sample_size...) to (nb_augms * batch_size, sample_size...)
			batch_u_augm = merge_first_dimension(batch_u_augm)

			batch_w = torch.cat((batch_s_augm, batch_u_augm))
			labels_w = torch.cat((labels_s, labels_u_guessed_repeated))

			# Shuffle batch and labels
			batch_w, labels_w = same_shuffle([batch_w, labels_w])

			len_s = len(batch_s_augm)
			batch_s_mixed, labels_s_mixed = self.mixup_mixer(
				batch_s_augm, labels_s, batch_w[:len_s], labels_w[:len_s])
			batch_u_mixed, labels_u_mixed = self.mixup_mixer(
				batch_u_augm, labels_u_guessed_repeated, batch_w[len_s:], labels_w[len_s:])

			return batch_s_mixed, labels_s_mixed, batch_u_mixed, labels_u_mixed
