import torch

from torch import Tensor
from torch.nn import BCELoss
from typing import Callable

from dcase2020_task4.util.utils_match import cross_entropy, binarize_onehot_labels


class FixMatchLossMultiHotV4(Callable):
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

		self.criterion_count = cross_entropy

	@staticmethod
	def from_edict(hparams) -> 'FixMatchLossMultiHotV4':
		return FixMatchLossMultiHotV4(hparams.lambda_u, hparams.threshold_mask, hparams.threshold_multihot)

	def __call__(
		self,
		s_pred_weak_augm_weak: Tensor,
		s_labels_weak: Tensor,
		u_pred_weak_augm_weak: Tensor,
		u_pred_weak_augm_strong: Tensor,
		u_labels_weak_guessed: Tensor,
		s_pred_count: Tensor,
		s_labels_count: Tensor,
		u_pred_count_augm_weak: Tensor,
		u_pred_count_augm_strong: Tensor,
	) -> (Tensor, Tensor, Tensor):
		# Supervised loss
		loss_s = self.criterion_s(s_pred_weak_augm_weak, s_labels_weak)
		loss_s = loss_s.mean()

		loss_sc = self.criterion_count(s_pred_count, s_labels_count)

		# Unsupervised loss
		u_counts = u_pred_count_augm_weak.argmax(dim=1)
		u_pred_sorted = u_pred_weak_augm_weak.sort(dim=1, descending=True)[0]
		mean_values = [
			pred[:count].mean() if count > 0 else 0.0
			for pred, count in zip(u_pred_sorted, u_counts)
		]
		mean_values = torch.as_tensor(mean_values).cuda()

		mask = (mean_values > self.threshold_mask).float()
		loss_u = self.criterion_u(u_pred_weak_augm_strong, u_labels_weak_guessed)
		loss_u *= mask
		loss_u = loss_u.mean()

		loss_uc = self.criterion_count(u_pred_count_augm_strong, binarize_onehot_labels(u_pred_count_augm_weak))
		loss_uc *= mask
		loss_uc = loss_uc.mean()

		loss = loss_s + self.lambda_u * loss_u + loss_sc + loss_uc

		return loss, loss_s, loss_u
