import torch

from torch import Tensor
from torch.nn import Module
from typing import Callable

from dcase2020_task4.mixup.mixer import MixUpMixer
from dcase2020_task4.util.utils_match import same_shuffle, sharpen, merge_first_dimension, sharpen_multi


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
		sharpen_threshold_multihot: float = 0.5,
	):
		self.model = model
		self.acti_fn = acti_fn
		self.nb_augms = nb_augms
		self.sharpen_temp = sharpen_temp
		self.mixup_mixer = MixUpMixer(alpha=mixup_alpha, apply_max=True)
		self.mode = mode
		self.sharpen_threshold_multihot = sharpen_threshold_multihot

	def __call__(self, s_batch_augm: Tensor, s_label: Tensor, u_batch_augms: Tensor) -> (Tensor, Tensor, Tensor, Tensor):
		return self.mix(s_batch_augm, s_label, u_batch_augms)

	def mix(self, s_batch_augm: Tensor, s_label: Tensor, u_batch_augms: Tensor) -> (Tensor, Tensor, Tensor, Tensor):
		"""
			s_batch_augm of size (bsize, feat_size, ...)
			s_labels_weak of size (bsize, label_size)
			u_batch_augms of size (nb_augms, bsize, feat_size, ...)
		"""
		with torch.no_grad():
			# Compute guessed label
			u_logits_augms = torch.zeros([self.nb_augms] + list(s_label.size())).cuda()
			for k in range(self.nb_augms):
				u_logits_augms[k] = self.model(u_batch_augms[k])
			u_pred_augms = self.acti_fn(u_logits_augms, dim=2)
			u_label_guessed = u_pred_augms.mean(dim=0)

			if self.mode == "onehot":
				u_label_guessed = sharpen(u_label_guessed, self.sharpen_temp, dim=1)
			elif self.mode == "multihot":
				u_label_guessed = sharpen_multi(u_label_guessed, self.sharpen_temp, self.sharpen_threshold_multihot)

			repeated_size = [self.nb_augms] + [1] * (len(u_label_guessed.size()) - 1)
			labels_u_guessed_repeated = u_label_guessed.repeat(repeated_size)
			u_batch_augms = merge_first_dimension(u_batch_augms)

			w_batch = torch.cat((s_batch_augm, u_batch_augms))
			w_label = torch.cat((s_label, labels_u_guessed_repeated))

			# Shuffle batch and labels
			w_batch, w_label = same_shuffle([w_batch, w_label])

			len_s = len(s_batch_augm)
			batch_s_mixed, labels_s_mixed = self.mixup_mixer(
				s_batch_augm, s_label, w_batch[:len_s], w_label[:len_s])
			batch_u_mixed, labels_u_mixed = self.mixup_mixer(
				u_batch_augms, labels_u_guessed_repeated, w_batch[len_s:], w_label[len_s:])

			return batch_s_mixed, labels_s_mixed, batch_u_mixed, labels_u_mixed
