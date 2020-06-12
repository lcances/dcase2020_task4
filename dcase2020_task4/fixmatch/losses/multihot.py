from torch import Tensor
from torch.nn import BCELoss

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
		max_values, _ = u_pred_weak_augm_weak.max(dim=1)
		mask = (max_values > self.threshold_mask).float()
		loss_u = self.criterion_u(u_pred_weak_augm_strong, u_labels_weak_guessed)
		loss_u *= mask
		loss_u = loss_u.mean()

		loss = loss_s + self.lambda_u * loss_u

		return loss, loss_s, loss_u


class FixMatchLossMultiHotV2(FixMatchLossABC):
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
	def from_edict(hparams) -> 'FixMatchLossMultiHotV2':
		return FixMatchLossMultiHotV2(hparams.lambda_u, hparams.threshold_mask, hparams.threshold_multihot)

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
		mean_values = u_pred_weak_augm_weak.mean(dim=1)
		mask = (mean_values > self.threshold_mask).float()
		loss_u = self.criterion_u(u_pred_weak_augm_strong, u_labels_weak_guessed)
		loss_u *= mask
		loss_u = loss_u.mean()

		loss = loss_s + self.lambda_u * loss_u

		return loss, loss_s, loss_u


class FixMatchLossMultiHotV3(FixMatchLossABC):
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
	def from_edict(hparams) -> 'FixMatchLossMultiHotV3':
		return FixMatchLossMultiHotV3(hparams.lambda_u, hparams.threshold_mask, hparams.threshold_multihot)

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
		mean_values = (u_pred_weak_augm_weak * u_labels_weak_guessed).sum(dim=1) / u_labels_weak_guessed.sum(dim=1)
		mask = (mean_values > self.threshold_mask).float()
		loss_u = self.criterion_u(u_pred_weak_augm_strong, u_labels_weak_guessed)
		loss_u *= mask
		loss_u = loss_u.mean()

		loss = loss_s + self.lambda_u * loss_u

		return loss, loss_s, loss_u


FixMatchLossMultiHot = FixMatchLossMultiHotV2
