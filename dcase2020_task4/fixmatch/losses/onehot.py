from torch import Tensor

from dcase2020_task4.fixmatch.losses.abc import FixMatchLossABC
from dcase2020_task4.util.utils_match import cross_entropy


class FixMatchLossOneHot(FixMatchLossABC):
	def __init__(
		self,
		lambda_u: float = 1.0,
		threshold_mask: float = 0.95,
	):
		self.lambda_u = lambda_u
		self.threshold_mask = threshold_mask

		self.criterion_s = cross_entropy
		self.criterion_u = cross_entropy

	@staticmethod
	def from_edict(hparams) -> 'FixMatchLossOneHot':
		return FixMatchLossOneHot(hparams.lambda_u, hparams.threshold_mask)

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
		max_values, _ = u_pred_weak_augm_weak.max(dim=1)
		mask = (max_values > self.threshold_mask).float()
		loss_u = self.criterion_u(u_pred_weak_augm_strong, u_labels_weak_guessed)
		loss_u *= mask
		loss_u = loss_u.mean()

		loss = loss_s + self.lambda_u * loss_u

		return loss, loss_s, loss_u
