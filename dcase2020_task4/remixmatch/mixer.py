import torch
from torch import Tensor
from torch.nn import Module
from typing import Callable

from dcase2020_task4.util.avg_distributions import AvgDistributions
from dcase2020_task4.mixup.mixers.monolabel import MixUpMixer
from dcase2020_task4.util.utils_match import normalize, same_shuffle, sharpen, merge_first_dimension, sharpen_multi


class ReMixMatchMixer(Callable):
	def __init__(
		self,
		model: Module,
		acti_fn: Callable,
		distributions: AvgDistributions,
		nb_augms_strong: int = 2,
		sharpen_temp: float = 0.5,
		mixup_alpha: float = 0.75,
		mode: str = "onehot",
		sharpen_threshold_multihot: float = 0.5,
	):
		self.model = model
		self.acti_fn = acti_fn
		self.distributions = distributions
		self.nb_augms_strong = nb_augms_strong
		self.sharpen_temp = sharpen_temp
		self.mode = mode
		self.sharpen_threshold_multihot = sharpen_threshold_multihot

		self.mixup_mixer = MixUpMixer(alpha=mixup_alpha, apply_max=True)

	def __call__(
		self, s_batch_strong: Tensor, s_label: Tensor, u_batch_weak: Tensor, u_batch_strongs: Tensor
	) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor):
		return self.mix(s_batch_strong, s_label, u_batch_weak, u_batch_strongs)

	def mix(
		self, s_batch_strong: Tensor, s_label: Tensor, u_batch_weak: Tensor, u_batch_strongs: Tensor
	) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor):
		"""
			s_batch_strong of size (bsize, feat_size, ...)
			s_labels_weak of size (bsize, label_size)
			u_batch_weak of size (bsize, feat_size, ...)
			u_batch_strongs of size (nb_augms, bsize, feat_size, ...)
		"""
		with torch.no_grad():
			# Compute guessed label
			u_logits_weak = self.model(u_batch_weak)
			u_label_guessed = self.acti_fn(u_logits_weak, dim=1)
			u_label_guessed = self.distributions.apply_distribution_alignment(u_label_guessed, dim=1)

			if self.mode == "onehot":
				u_label_guessed = sharpen(u_label_guessed, self.sharpen_temp, dim=1)
			elif self.mode == "multihot":
				u_label_guessed = sharpen_multi(u_label_guessed, self.sharpen_temp, self.sharpen_threshold_multihot)
			else:
				raise RuntimeError("Invalid argument \"mode = %s\". Use %s." % (self.mode, " or ".join(("onehot", "multihot"))))

			# Get strongly augmented batch "batch_u1"
			batch_u1 = u_batch_strongs[0, :].clone()
			labels_u1 = u_label_guessed.clone()

			repeated_size = [self.nb_augms_strong] + [1] * (len(u_label_guessed.size()) - 1)
			labels_u_guessed_repeated = u_label_guessed.repeat(repeated_size)
			u_batch_strongs = merge_first_dimension(u_batch_strongs)

			# Concatenate strongly and weakly augmented data from batch_u
			batch_u_cat = torch.cat((u_batch_strongs, u_batch_weak), dim=0)
			labels_u_cat = torch.cat((labels_u_guessed_repeated, u_label_guessed), dim=0)

			batch_w = torch.cat((s_batch_strong, batch_u_cat), dim=0)
			labels_w = torch.cat((s_label, labels_u_cat), dim=0)

			# Shuffle batch and labels
			batch_w, labels_w = same_shuffle([batch_w, labels_w])

			len_s = len(s_batch_strong)
			batch_s_mixed, labels_s_mixed = self.mixup_mixer(
				s_batch_strong, s_label, batch_w[:len_s], labels_w[:len_s]
			)
			batch_u_mixed, labels_u_mixed = self.mixup_mixer(
				batch_u_cat, labels_u_cat, batch_w[len_s:], labels_w[len_s:]
			)

			return batch_s_mixed, labels_s_mixed, batch_u_mixed, labels_u_mixed, batch_u1, labels_u1
