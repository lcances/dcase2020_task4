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
		acti_fn: Callable,
		nb_augms: int = 2,
		sharpen_temp: float = 0.5,
		mixup_alpha: float = 0.75,
		mode: str = "onehot",
	):
		self.model = model
		self.acti_fn = acti_fn
		self.nb_augms = nb_augms
		self.sharpen_temp = sharpen_temp
		self.mixup_mixer = MixUpMixer(alpha=mixup_alpha, apply_max=True)
		self.mode = mode

	def __call__(self, batch_s_augm: Tensor, labels_s: Tensor, batch_u_augms: Tensor) -> (Tensor, Tensor, Tensor, Tensor):
		return self.mix(batch_s_augm, labels_s, batch_u_augms)

	def mix(self, batch_s_augm: Tensor, labels_s: Tensor, batch_u_augms: Tensor) -> (Tensor, Tensor, Tensor, Tensor):
		"""
			batch_s_augm of size (bsize, feat_size, ...)
			labels_s of size (bsize, label_size)
			batch_u_augms of size (nb_augms, bsize, feat_size, ...)
		"""
		with torch.no_grad():
			# Compute guessed label
			logits_u_augm = torch.zeros([self.nb_augms] + list(labels_s.size())).cuda()
			for k in range(self.nb_augms):
				logits_u_augm[k] = self.model(batch_u_augms[k])
			predictions_u_augm = self.acti_fn(logits_u_augm, dim=2)
			labels_u_guessed = predictions_u_augm.mean(dim=0)
			if self.mode == "onehot":
				labels_u_guessed = sharpen(labels_u_guessed, self.sharpen_temp, dim=1)

			repeated_size = [self.nb_augms] + [1] * (len(labels_u_guessed.size()) - 1)
			labels_u_guessed_repeated = labels_u_guessed.repeat(repeated_size)
			batch_u_augms = merge_first_dimension(batch_u_augms)

			batch_w = torch.cat((batch_s_augm, batch_u_augms))
			labels_w = torch.cat((labels_s, labels_u_guessed_repeated))

			# Shuffle batch and labels
			batch_w, labels_w = same_shuffle([batch_w, labels_w])

			len_s = len(batch_s_augm)
			batch_s_mixed, labels_s_mixed = self.mixup_mixer(
				batch_s_augm, labels_s, batch_w[:len_s], labels_w[:len_s])
			batch_u_mixed, labels_u_mixed = self.mixup_mixer(
				batch_u_augms, labels_u_guessed_repeated, batch_w[len_s:], labels_w[len_s:])

			return batch_s_mixed, labels_s_mixed, batch_u_mixed, labels_u_mixed
