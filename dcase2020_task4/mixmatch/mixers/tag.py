import torch

from torch import Tensor
from typing import Callable

from dcase2020_task4.mixup.mixers.abc import MixUpMixerTagABC
from dcase2020_task4.util.utils_match import same_shuffle, merge_first_dimension


class MixMatchMixer(Callable):
	"""
		MixMatch class.
		Store hyperparameters and apply mixmatch_fn with call() or mix().
	"""
	def __init__(self, mixup_mixer: MixUpMixerTagABC, shuffle_s_with_u: bool = True):
		self.mixup_mixer = mixup_mixer
		self.shuffle_s_with_u = shuffle_s_with_u

	def __call__(
		self, s_batch_augm: Tensor, s_label: Tensor, u_batch_augms: Tensor, u_label_guessed: Tensor
	) -> (Tensor, Tensor, Tensor, Tensor):
		return self.mix(s_batch_augm, s_label, u_batch_augms, u_label_guessed)

	def mix(
		self, s_batch_augm: Tensor, s_label: Tensor, u_batch_augms: Tensor, u_label_guessed: Tensor
	) -> (Tensor, Tensor, Tensor, Tensor):
		"""
			s_batch_augm of size (bsize, feat_size, ...)
			s_label_weak of size (bsize, label_size)
			u_batch_augms of size (nb_augms, bsize, feat_size, ...)
		"""
		with torch.no_grad():
			nb_augms = u_batch_augms.shape[0]
			repeated_size = [nb_augms] + [1] * (len(u_label_guessed.size()) - 1)
			labels_u_guessed_repeated = u_label_guessed.repeat(repeated_size)
			u_batch_augms = merge_first_dimension(u_batch_augms)

			if self.shuffle_s_with_u:
				w_batch = torch.cat((s_batch_augm, u_batch_augms))
				w_label = torch.cat((s_label, labels_u_guessed_repeated))

				# Shuffle batch and labels
				w_batch, w_label = same_shuffle([w_batch, w_label])
			else:
				s_batch_augm, s_label = same_shuffle([s_batch_augm, s_label])
				u_batch_augms, labels_u_guessed_repeated = same_shuffle([u_batch_augms, labels_u_guessed_repeated])

				w_batch = torch.cat((s_batch_augm, u_batch_augms))
				w_label = torch.cat((s_label, labels_u_guessed_repeated))

			len_s = len(s_batch_augm)
			s_batch_mixed, s_label_mixed = self.mixup_mixer(
				s_batch_augm, s_label, w_batch[:len_s], w_label[:len_s])
			u_batch_mixed, u_label_mixed = self.mixup_mixer(
				u_batch_augms, labels_u_guessed_repeated, w_batch[len_s:], w_label[len_s:])

			return s_batch_mixed, s_label_mixed, u_batch_mixed, u_label_mixed


def test():
	batch = torch.as_tensor([
		[1, 1, 1, 1],
		[2, 2, 2, 2],
		[3, 3, 3, 3.]
	])

	nb_augms = 2
	batch_augms = torch.as_tensor([
		(batch + 0.1).tolist(), (batch + 0.2).tolist()
	])

	label = torch.as_tensor([10, 20, 30])

	repeated_size = [nb_augms] + [1] * (len(label.size()) - 1)
	label_repeated = label.repeat(repeated_size)
	batch_augms_merged = merge_first_dimension(batch_augms)

	w_batch = torch.cat((batch, batch_augms_merged))
	w_label = torch.cat((label, label_repeated))

	w_batch, w_label = same_shuffle([w_batch, w_label])

	print(w_batch)
	print(w_label)


if __name__ == "__main__":
	test()
