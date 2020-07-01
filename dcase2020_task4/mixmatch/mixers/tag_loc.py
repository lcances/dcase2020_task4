import torch

from torch import Tensor
from torch.nn import Module
from typing import Callable

from dcase2020_task4.mixup.mixers.tag_loc import MixUpMixerLoc
from dcase2020_task4.util.utils_match import same_shuffle, merge_first_dimension, sharpen_multi


class MixMatchMixerMultiHotLoc(Callable):
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
		sharpen_threshold_multihot: float = 0.5,
	):
		self.model = model
		self.acti_fn = acti_fn
		self.nb_augms = nb_augms
		self.sharpen_temp = sharpen_temp
		self.mixup_mixer = MixUpMixerLoc(alpha=mixup_alpha, apply_max=True)
		self.sharpen_threshold_multihot = sharpen_threshold_multihot

	def __call__(self, s_batch_augm: Tensor, s_label_weak: Tensor, s_label_strong: Tensor, u_batch_augms: Tensor) -> (Tensor, Tensor, Tensor, Tensor):
		return self.mix(s_batch_augm, s_label_weak, s_label_strong, u_batch_augms)

	def mix(
		self, s_batch_augm: Tensor, s_label_weak: Tensor, s_label_strong: Tensor, u_batch_augms: Tensor
	) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor):
		with torch.no_grad():
			# Compute guessed label
			u_logits_weak_augms = torch.zeros([self.nb_augms] + list(s_label_weak.size())).cuda()
			u_logits_strong_augms = torch.zeros([self.nb_augms] + list(s_label_strong.size())).cuda()
			for k in range(self.nb_augms):
				u_logits_weak_augms[k], u_logits_strong_augms[k] = self.model(u_batch_augms[k])
			u_pred_weak_augms = self.acti_fn(u_logits_weak_augms, dim=2)
			u_pred_strong_augms = self.acti_fn(u_logits_strong_augms, dim=2)

			u_label_weak_guessed = u_pred_weak_augms.mean(dim=0)
			u_label_strong_guessed = u_pred_strong_augms.mean(dim=0)

			u_label_weak_guessed = sharpen_multi(
				u_label_weak_guessed, self.sharpen_temp, self.sharpen_threshold_multihot)
			u_label_strong_guessed = sharpen_multi(
				u_label_strong_guessed, self.sharpen_temp, self.sharpen_threshold_multihot)

			repeated_size_weak = [self.nb_augms] + [1] * (len(u_label_weak_guessed.size()) - 1)
			labels_u_weak_guessed_repeated = u_label_weak_guessed.repeat(repeated_size_weak)

			repeated_size_strong = [self.nb_augms] + [1] * (len(u_label_strong_guessed.size()) - 1)
			labels_u_strong_guessed_repeated = u_label_strong_guessed.repeat(repeated_size_strong)

			u_batch_augms = merge_first_dimension(u_batch_augms)

			w_batch = torch.cat((s_batch_augm, u_batch_augms))
			w_label_weak = torch.cat((s_label_weak, labels_u_weak_guessed_repeated))
			w_label_strong = torch.cat((s_label_strong, labels_u_strong_guessed_repeated))

			# Shuffle batch and labels
			w_batch, w_label_weak, w_label_strong = same_shuffle([w_batch, w_label_weak, w_label_strong])

			s_len = len(s_batch_augm)
			s_batch_mixed, (s_label_weak_mixed, s_label_strong_mixed) = self.mixup_mixer(
				s_batch_augm, [s_label_weak, s_label_strong],
				w_batch[:s_len], [w_label_weak[:s_len], w_label_strong[:s_len]]
			)
			u_batch_mixed, (u_label_weak_mixed, u_label_strong_mixed) = self.mixup_mixer(
				u_batch_augms, [labels_u_weak_guessed_repeated, labels_u_strong_guessed_repeated],
				w_batch[s_len:], [w_label_weak[s_len:], w_label_strong[s_len:]]
			)

			return s_batch_mixed, s_label_weak_mixed, s_label_strong_mixed, u_batch_mixed, u_label_weak_mixed, u_label_strong_mixed
