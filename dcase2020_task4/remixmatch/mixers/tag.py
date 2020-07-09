import torch
from torch import Tensor
from typing import Callable

from dcase2020_task4.mixup.mixers.abc import MixUpMixerTagABC
from dcase2020_task4.util.utils_match import same_shuffle, merge_first_dimension


class ReMixMatchMixer(Callable):
	def __init__(self, mixup_mixer: MixUpMixerTagABC, shuffle_s_with_u: bool = True):
		self.mixup_mixer = mixup_mixer
		self.shuffle_s_with_u = shuffle_s_with_u

	def __call__(
		self,
		s_batch_strong: Tensor, s_label: Tensor,
		u_batch_weak: Tensor, u_batch_strongs: Tensor, u_label_guessed: Tensor,
	) -> (Tensor, Tensor, Tensor, Tensor):
		"""
			s_batch_strong of size (bsize, feat_size, ...)
			s_labels_weak of size (bsize, label_size)
			u_batch_weak of size (bsize, feat_size, ...)
			u_batch_strongs of size (nb_augms_strong, bsize, feat_size, ...)
		"""
		return self.mix(s_batch_strong, s_label, u_batch_weak, u_batch_strongs, u_label_guessed)

	def mix(
		self,
		s_batch_strong: Tensor, s_label: Tensor,
		u_batch_weak: Tensor, u_batch_strongs: Tensor, u_label_guessed: Tensor,
	) -> (Tensor, Tensor, Tensor, Tensor):
		"""
			s_batch_strong of size (bsize, feat_size, ...)
			s_labels_weak of size (bsize, label_size)
			u_batch_weak of size (bsize, feat_size, ...)
			u_batch_strongs of size (nb_augms_strong, bsize, feat_size, ...)
		"""
		with torch.no_grad():
			nb_augms_strong = u_batch_strongs.shape[0]
			repeated_size = [nb_augms_strong] + [1] * (len(u_label_guessed.size()) - 1)
			u_label_guessed_repeated = u_label_guessed.repeat(repeated_size)
			u_batch_strongs = merge_first_dimension(u_batch_strongs)

			# Concatenate strongly and weakly augmented data from batch_u
			u_batch_cat = torch.cat((u_batch_strongs, u_batch_weak), dim=0)
			u_label_cat = torch.cat((u_label_guessed_repeated, u_label_guessed), dim=0)

			if self.shuffle_s_with_u:
				w_batch = torch.cat((s_batch_strong, u_batch_cat), dim=0)
				w_label = torch.cat((s_label, u_label_cat), dim=0)

				# Shuffle batch and labels
				w_batch, w_label = same_shuffle([w_batch, w_label])
			else:
				s_batch_strong, s_label = same_shuffle([s_batch_strong, s_label])
				u_batch_cat, u_label_cat = same_shuffle([u_batch_cat, u_label_cat])

				w_batch = torch.cat((s_batch_strong, u_batch_cat), dim=0)
				w_label = torch.cat((s_label, u_label_cat), dim=0)

			len_s = len(s_batch_strong)
			batch_s_mixed, labels_s_mixed = self.mixup_mixer(
				s_batch_strong, s_label, w_batch[:len_s], w_label[:len_s]
			)
			batch_u_mixed, labels_u_mixed = self.mixup_mixer(
				u_batch_cat, u_label_cat, w_batch[len_s:], w_label[len_s:]
			)

			return batch_s_mixed, labels_s_mixed, batch_u_mixed, labels_u_mixed
