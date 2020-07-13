from torch import Tensor
from typing import Optional

from dcase2020_task4.fixmatch.losses.abc import FixMatchLossTagABC
from dcase2020_task4.util.utils_match import cross_entropy


class FixMatchLossOneHot(FixMatchLossTagABC):
	def __init__(
		self,
		lambda_u: float = 1.0,
		threshold_confidence: float = 0.95,
	):
		self.lambda_u = lambda_u
		self.threshold_confidence = threshold_confidence

		self.criterion_s = cross_entropy
		self.criterion_u = cross_entropy
		self.last_mask = None

	@staticmethod
	def from_edict(hparams) -> 'FixMatchLossOneHot':
		return FixMatchLossOneHot(hparams.lambda_u, hparams.threshold_confidence)

	def __call__(
		self,
		s_pred_augm_weak: Tensor,
		s_labels: Tensor,
		u_pred_augm_weak: Tensor,
		u_pred_augm_strong: Tensor,
		u_labels_guessed: Tensor,
	) -> (Tensor, Tensor, Tensor):
		# Supervised loss
		loss_s = self.criterion_s(s_pred_augm_weak, s_labels)
		loss_s = loss_s.mean()

		# Unsupervised loss
		mask = self.confidence_mask(u_pred_augm_weak, dim=1)
		loss_u = self.criterion_u(u_pred_augm_strong, u_labels_guessed)
		loss_u *= mask
		loss_u = loss_u.mean()

		loss = loss_s + self.lambda_u * loss_u
		self.last_mask = mask.detach()

		return loss, loss_s, loss_u

	def confidence_mask(self, pred_weak: Tensor, dim: int) -> Tensor:
		max_values, _ = pred_weak.max(dim=dim)
		return (max_values > self.threshold_confidence).float()

	def get_last_mask(self) -> Optional[Tensor]:
		return self.last_mask

	def get_lambda_u(self) -> float:
		return self.lambda_u
