import torch

from torch import Tensor
from typing import Callable

from dcase2020_task4.util.utils_match import same_shuffle, merge_first_dimension


class MixMatchMixerMultiHotLoc(Callable):
	"""
		MixMatch mixer class for multihot with localization.
	"""
	def __init__(self, mixup_mixer: Callable):
		self.mixup_mixer = mixup_mixer

	def __call__(
		self,
		s_batch_augm: Tensor, s_label_weak: Tensor, s_label_strong: Tensor,
		u_batch_augms: Tensor, u_label_weak_guessed: Tensor, u_label_strong_guessed: Tensor,
	) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor):
		return self.mix(
			s_batch_augm, s_label_weak, s_label_strong, u_batch_augms, u_label_weak_guessed, u_label_strong_guessed)

	def mix(
		self,
		s_batch_augm: Tensor, s_label_weak: Tensor, s_label_strong: Tensor,
		u_batch_augms: Tensor, u_label_weak_guessed: Tensor, u_label_strong_guessed: Tensor,
	) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor):
		"""
			s_batch_augm of size (bsize, feat_size)
			s_label_weak of size (bsize, nb_classes)
			s_label_strong of size (bsize, nb_classes, feat_size)

			u_batch_augms of size (nb_augms, bsize, feat_size)
			u_label_weak_guessed of size (bsize, nb_classes)
			u_label_strong_guessed of size (bsize, nb_classes, feat_size)
		"""
		with torch.no_grad():
			nb_augms = u_batch_augms.shape[0]
			repeated_size_weak = [nb_augms] + [1] * (len(u_label_weak_guessed.size()) - 1)
			labels_u_weak_guessed_repeated = u_label_weak_guessed.repeat(repeated_size_weak)

			repeated_size_strong = [nb_augms] + [1] * (len(u_label_strong_guessed.size()) - 1)
			labels_u_strong_guessed_repeated = u_label_strong_guessed.repeat(repeated_size_strong)

			u_batch_augms = merge_first_dimension(u_batch_augms)

			# TODO : add shuffle_s_and_u option
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
