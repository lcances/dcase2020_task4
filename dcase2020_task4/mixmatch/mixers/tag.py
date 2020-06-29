import torch

from torch import Tensor
from torch.nn import Module
from typing import Callable, Optional

from dcase2020_task4.mixup.mixers.monolabel import MixUpMixer
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
		sharpen_threshold_multihot: Optional[float] = None,
	):
		self.model = model
		self.acti_fn = acti_fn
		self.nb_augms = nb_augms
		self.sharpen_temp = sharpen_temp
		self.mixup_mixer = MixUpMixer(alpha=mixup_alpha, apply_max=True)
		self.mode = mode
		self.sharpen_threshold_multihot = sharpen_threshold_multihot

		if self.mode == "multihot" and self.sharpen_threshold_multihot is None:
			raise RuntimeError("Multihot Sharpen threshold cannot be None in multihot mode.")

	def __call__(self, s_batch_augm: Tensor, s_label: Tensor, u_batch_augms: Tensor) -> (Tensor, Tensor, Tensor, Tensor):
		return self.mix(s_batch_augm, s_label, u_batch_augms)

	def mix(self, s_batch_augm: Tensor, s_label: Tensor, u_batch_augms: Tensor) -> (Tensor, Tensor, Tensor, Tensor):
		"""
			s_batch_augm of size (bsize, feat_size, ...)
			s_label_weak of size (bsize, label_size)
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
			else:
				raise RuntimeError("Invalid argument \"mode = %s\". Use %s." % (self.mode, " or ".join(("onehot", "multihot"))))

			repeated_size = [self.nb_augms] + [1] * (len(u_label_guessed.size()) - 1)
			labels_u_guessed_repeated = u_label_guessed.repeat(repeated_size)
			u_batch_augms = merge_first_dimension(u_batch_augms)

			w_batch = torch.cat((s_batch_augm, u_batch_augms))
			w_label = torch.cat((s_label, labels_u_guessed_repeated))

			# Shuffle batch and labels
			w_batch, w_label = same_shuffle([w_batch, w_label])

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
