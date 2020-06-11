import torch
from torch import Tensor
from torch.nn import Module
from typing import Callable

from dcase2020_task4.remixmatch.model_distributions import ModelDistributions
from dcase2020_task4.mixup.mixer import MixUpMixer
from dcase2020_task4.util.utils_match import normalize, same_shuffle, sharpen, merge_first_dimension


class ReMixMatchMixer(Callable):
	def __init__(
		self,
		model: Module,
		acti_fn: Callable,
		distributions: ModelDistributions,
		nb_augms_strong: int,
		sharpen_temp: float,
		mixup_alpha: float,
		mode: str = "onehot",
	):
		self.model = model
		self.acti_fn = acti_fn
		self.distributions = distributions
		self.nb_augms_strong = nb_augms_strong
		self.sharpen_temp = sharpen_temp
		self.mode = mode

		self.mixup_mixer = MixUpMixer(alpha=mixup_alpha, apply_max=True)

	def __call__(
		self, batch_s_strong: Tensor, labels_s: Tensor, batch_u_weak: Tensor, batch_u_strongs: Tensor
	) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor):
		return self.mix(batch_s_strong, labels_s, batch_u_weak, batch_u_strongs)

	def mix(
		self, batch_s_strong: Tensor, labels_s: Tensor, batch_u_weak: Tensor, batch_u_strongs: Tensor
	) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor):
		"""
			batch_s_strong of size (bsize, feat_size, ...)
			s_labels_weak of size (bsize, label_size)
			batch_u_weak of size (bsize, feat_size, ...)
			batch_u_strongs of size (nb_augms, bsize, feat_size, ...)
		"""
		with torch.no_grad():
			# Compute guessed label
			logits_u_weak = self.model(batch_u_weak)
			labels_u_guessed = self.acti_fn(logits_u_weak, dim=1)
			labels_u_guessed *= self.distributions.get_mean_pred("labeled") / self.distributions.get_mean_pred("unlabeled")
			if self.mode == "onehot":
				labels_u_guessed = normalize(labels_u_guessed, dim=1)
				labels_u_guessed = sharpen(labels_u_guessed, self.sharpen_temp, dim=1)

			# Get strongly augmented batch "batch_u1"
			batch_u1 = batch_u_strongs[0, :].clone()
			labels_u1 = labels_u_guessed.clone()

			repeated_size = [self.nb_augms_strong] + [1] * (len(labels_u_guessed.size()) - 1)
			labels_u_guessed_repeated = labels_u_guessed.repeat(repeated_size)
			batch_u_strongs = merge_first_dimension(batch_u_strongs)

			# Concatenate strongly and weakly augmented data from batch_u
			batch_u_cat = torch.cat((batch_u_strongs, batch_u_weak), dim=0)
			labels_u_cat = torch.cat((labels_u_guessed_repeated, labels_u_guessed), dim=0)

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
