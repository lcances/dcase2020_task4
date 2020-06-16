from torch import Tensor
from torch.nn import BCELoss
from typing import Union

from dcase2020_task4.fixmatch.losses.abc import FixMatchLossABC


class FixMatchLossMultiHotV1(FixMatchLossABC):
	def __init__(
		self,
		lambda_u: float = 1.0,
		threshold_mask: float = 0.5,
		threshold_multihot: float = 0.5,
	):
		self.lambda_u = lambda_u
		self.threshold_mask = threshold_mask
		self.threshold_multihot = threshold_multihot

		self.criterion_s = BCELoss(reduction="none")
		# Note : we need a loss per example and not a mean reduction on all loss
		self.criterion_u = lambda pred, labels: BCELoss(reduction="none")(pred, labels).mean(dim=1)

	@staticmethod
	def from_edict(hparams) -> 'FixMatchLossMultiHotV1':
		return FixMatchLossMultiHotV1(hparams.lambda_u, hparams.threshold_mask, hparams.threshold_multihot)

	def __call__(
		self,
		s_pred_weak_augm_weak: Tensor,
		s_labels_weak: Tensor,
		u_pred_weak_augm_weak: Tensor,
		u_pred_weak_augm_strong: Tensor,
		u_labels_weak_guessed: Tensor,
	) -> (Tensor, Tensor, Tensor):
		# Supervised loss
		loss_s = self.criterion_s(s_pred_weak_augm_weak, s_labels_weak)
		loss_s = loss_s.mean()

		# Unsupervised loss
		mask = self.get_confidence_mask(u_pred_weak_augm_weak, dim=1)
		loss_u = self.criterion_u(u_pred_weak_augm_strong, u_labels_weak_guessed)
		loss_u *= mask
		loss_u = loss_u.mean()

		loss = loss_s + self.lambda_u * loss_u

		return loss, loss_s, loss_u

	def get_confidence_mask(self, pred: Tensor, dim: Union[int, tuple]) -> Tensor:
		means, _ = pred.max(dim=dim)
		return (means > self.threshold_mask).float()
