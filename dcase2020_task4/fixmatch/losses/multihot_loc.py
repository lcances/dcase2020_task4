import torch

from torch import Tensor
from torch.nn import BCELoss
from typing import Callable, Union


class FixMatchLossMultiHotLoc(Callable):
	def __init__(
		self,
		lambda_u: float = 1.0,
		threshold_mask: float = 0.5,
		threshold_multihot: float = 0.5,
	):
		self.lambda_u = lambda_u
		self.threshold_mask = threshold_mask
		self.threshold_multihot = threshold_multihot

		self.criterion_s_weak = BCELoss(reduction="none")
		# Note : we need a loss per example and not a mean reduction on all loss
		self.criterion_u_weak = lambda pred, labels: BCELoss(reduction="none")(pred, labels).mean(dim=1)

		self.criterion_s_strong = BCELoss(reduction="none")
		self.criterion_u_strong = BCELoss(reduction="none")

	@staticmethod
	def from_edict(hparams) -> 'FixMatchLossMultiHotLoc':
		return FixMatchLossMultiHotLoc(hparams.lambda_u, hparams.threshold_mask, hparams.threshold_multihot)

	def __call__(
		self,
		s_pred_weak_augm_weak: Tensor, s_labels_weak: Tensor,
		u_pred_weak_augm_weak: Tensor, u_pred_weak_augm_strong: Tensor, u_labels_weak_guessed: Tensor,
		s_pred_strong_augm_weak: Tensor, s_labels_strong: Tensor,
		u_pred_strong_augm_weak: Tensor, u_pred_strong_augm_strong: Tensor, u_labels_strong_guessed: Tensor,
	) -> (Tensor, Tensor, Tensor, Tensor, Tensor):
		# Supervised weak loss
		loss_s_weak = self.criterion_s_weak(s_pred_weak_augm_weak, s_labels_weak)
		loss_s_weak = loss_s_weak.mean()

		# Supervised strong loss
		s_mask_has_strong = self.get_has_strong_mask(s_labels_strong)

		loss_s_strong = self.criterion_s_strong(s_pred_strong_augm_weak, s_labels_strong).mean(dim=(1, 2))
		loss_s_strong = s_mask_has_strong * loss_s_strong
		loss_s_strong = loss_s_strong.mean()

		# Unsupervised weak loss
		u_mask_confidence_weak = self.get_confidence_mask(u_pred_weak_augm_weak, dim=1)

		loss_u_weak = self.criterion_u_weak(u_pred_weak_augm_strong, u_labels_weak_guessed)
		loss_u_weak *= u_mask_confidence_weak
		loss_u_weak = loss_u_weak.mean()

		# Unsupervised strong loss
		u_mask_has_strong = self.get_has_strong_mask(u_labels_strong_guessed)
		u_mask_confidence_strong = self.get_confidence_mask(u_pred_strong_augm_weak, dim=(1, 2))

		loss_u_strong = self.criterion_u_strong(u_pred_strong_augm_strong, u_labels_strong_guessed).mean(dim=(1, 2))
		loss_u_strong = u_mask_has_strong * u_mask_confidence_strong * loss_u_strong
		loss_u_strong = loss_u_strong.mean()

		# Compute final loss
		loss = loss_s_weak + loss_s_strong + self.lambda_u * (loss_u_weak + loss_u_strong)

		return loss, loss_s_weak, loss_u_weak, loss_s_strong, loss_u_strong

	def get_has_strong_mask(self, labels_strong: Tensor) -> Tensor:
		return torch.clamp(labels_strong.sum(dim=(1, 2)), 0, 1)

	def get_confidence_mask(self, pred: Tensor, dim: Union[int, tuple]) -> Tensor:
		means = pred.mean(dim=dim)
		return (means > self.threshold_mask).float()

	def get_confidence_mask_v1(self, pred: Tensor, dim: Union[int, tuple]) -> Tensor:
		if type(dim) == int:
			maxes, _ = pred.max(dim=dim)
		else:
			maxes = pred.clone()
			for d in reversed(dim):
				maxes = maxes.max(dim=d)[0]
		return (maxes > self.threshold_mask).float()

	def get_confidence_mask_v3(self, pred: Tensor, labels: Tensor, dim: Union[int, tuple]) -> Tensor:
		means = (pred * labels).sum(dim=dim) / labels.sum(dim=dim)
		return (means > self.threshold_mask).float()


def test():
	batch_size = 16
	nb_classes = 10
	audio_size = 431

	s_pred_weak_augm_weak = torch.zeros(batch_size, nb_classes)
	s_labels_weak = torch.zeros(batch_size, nb_classes)
	u_pred_weak_augm_weak = torch.zeros(batch_size, nb_classes)
	u_pred_weak_augm_strong = torch.zeros(batch_size, nb_classes)
	u_labels_weak_guessed = torch.zeros(batch_size, nb_classes)

	s_pred_strong_augm_weak = torch.zeros(batch_size, nb_classes, audio_size)
	s_labels_strong = torch.zeros(batch_size, nb_classes, audio_size)
	u_pred_strong_augm_weak = torch.zeros(batch_size, nb_classes, audio_size)
	u_pred_strong_augm_strong = torch.zeros(batch_size, nb_classes, audio_size)
	u_labels_strong_guessed = torch.zeros(batch_size, nb_classes, audio_size)

	loss = FixMatchLossMultiHotLoc()

	loss, loss_s_weak, loss_u_weak, loss_s_strong, loss_u_strong = loss(
		s_pred_weak_augm_weak,
		s_labels_weak,
		u_pred_weak_augm_weak,
		u_pred_weak_augm_strong,
		u_labels_weak_guessed,
		s_pred_strong_augm_weak,
		s_labels_strong,
		u_pred_strong_augm_weak,
		u_pred_strong_augm_strong,
		u_labels_strong_guessed,
	)

	print("DEBUG: ", loss.shape, loss_s_weak.shape, loss_u_weak.shape, loss_s_strong.shape, loss_u_strong.shape)
	print("DEBUG: ", loss, loss_s_weak, loss_u_weak, loss_s_strong, loss_u_strong)


if __name__ == "__main__":
	test()