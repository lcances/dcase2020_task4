import torch

from argparse import Namespace
from torch import Tensor
from torch.nn import BCELoss
from typing import Union

from dcase2020_task4.fixmatch.losses.abc import FixMatchLossLocABC


class FixMatchLossMultiHotLocV3(FixMatchLossLocABC):
	""" FixMatch loss multi-hot (based on FixMatchLossMultiHotV3 class tagging). """

	def __init__(
		self,
		lambda_u: float = 1.0,
		threshold_confidence: float = 0.5,
		threshold_multihot: float = 0.5,
	):
		self.lambda_u = lambda_u
		self.threshold_confidence = threshold_confidence
		self.threshold_multihot = threshold_multihot

		self.criterion_s_weak = BCELoss(reduction="none")
		self.criterion_u_weak = BCELoss(reduction="none")

		self.criterion_s_strong = BCELoss(reduction="none")
		self.criterion_u_strong = BCELoss(reduction="none")

	@staticmethod
	def from_args(args: Namespace) -> 'FixMatchLossMultiHotLocV3':
		return FixMatchLossMultiHotLocV3(hparams.lambda_u, hparams.threshold_confidence, hparams.threshold_multihot)

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
		s_mask_strong = self.strong_mask(s_labels_strong)

		loss_s_strong = self.criterion_s_strong(s_pred_strong_augm_weak, s_labels_strong).mean(dim=(1, 2))
		loss_s_strong = s_mask_strong * loss_s_strong
		loss_s_strong = loss_s_strong.mean()

		# Unsupervised weak loss
		u_mask_confidence_weak = self.confidence_mask(u_pred_weak_augm_weak, u_labels_weak_guessed, dim=1)

		loss_u_weak = self.criterion_u_weak(u_pred_weak_augm_strong, u_labels_weak_guessed).mean(dim=1)
		loss_u_weak = loss_u_weak * u_mask_confidence_weak
		loss_u_weak = loss_u_weak.mean()

		# Unsupervised strong loss
		u_mask_strong = self.strong_mask(u_labels_strong_guessed)
		u_mask_confidence_strong = self.confidence_mask(u_pred_strong_augm_weak, u_labels_strong_guessed, dim=(1, 2))

		loss_u_strong = self.criterion_u_strong(u_pred_strong_augm_strong, u_labels_strong_guessed).mean(dim=(1, 2))
		loss_u_strong = loss_u_strong * u_mask_strong * u_mask_confidence_strong
		loss_u_strong = loss_u_strong.mean()

		# Compute final loss
		loss = loss_s_weak + loss_s_strong + self.lambda_u * (loss_u_weak + loss_u_strong)

		return loss, loss_s_weak, loss_u_weak, loss_s_strong, loss_u_strong

	def strong_mask(self, labels_strong: Tensor) -> Tensor:
		""" Strong labels of shape contains at least one "1". """
		return torch.clamp(labels_strong.sum(dim=(1, 2)), 0, 1)

	def confidence_mask(self, pred: Tensor, labels: Tensor, dim: Union[int, tuple]) -> Tensor:
		means = (pred * labels).sum(dim=dim) / labels.sum(dim=dim).clamp(min=1.0)
		return (means > self.threshold_confidence).float()

	def get_lambda_u(self) -> float:
		return self.lambda_u
