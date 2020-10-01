import torch

from argparse import Namespace
from torch import Tensor
from torch.nn import BCELoss
from typing import Callable, Optional

from dcase2020_task4.util.utils_match import cross_entropy, binarize_pred_to_onehot


class FixMatchLossMultiHotV4(Callable):
	def __init__(
		self,
		lambda_s: float = 1.0,
		lambda_u: float = 1.0,
		threshold_confidence: float = 0.95,
		threshold_multihot: float = 0.5,
	):
		self.lambda_s = lambda_s
		self.lambda_u = lambda_u
		self.threshold_confidence = threshold_confidence
		self.threshold_multihot = threshold_multihot

		self.criterion_s = BCELoss(reduction="none")
		self.criterion_u = BCELoss(reduction="none")
		self.criterion_count = cross_entropy
		self.last_mask = None

	@staticmethod
	def from_args(args: Namespace) -> 'FixMatchLossMultiHotV4':
		return FixMatchLossMultiHotV4(args.lambda_s, args.lambda_u, args.threshold_confidence, args.threshold_multihot)

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
	) -> (Tensor, Tensor, Tensor, Tensor, Tensor):
		# Supervised loss
		loss_s = self.criterion_s(s_pred_weak_augm_weak, s_labels_weak)
		loss_s = loss_s.mean()

		loss_sc = self.criterion_count(s_pred_count, s_labels_count)
		loss_sc = loss_sc.mean()

		# Unsupervised loss
		mask = self.get_confidence_mask(u_pred_weak_augm_weak, u_pred_count_augm_weak, dim=1)
		loss_u = self.criterion_u(u_pred_weak_augm_strong, u_labels_weak_guessed).mean(dim=1)
		loss_u *= mask
		loss_u = loss_u.mean()

		loss_uc = self.criterion_count(u_pred_count_augm_strong, binarize_pred_to_onehot(u_pred_count_augm_weak))
		loss_uc *= mask
		loss_uc = loss_uc.mean()

		loss = self.lambda_s * loss_s + self.lambda_u * loss_u + loss_sc + 0.1 * loss_uc
		self.last_mask = mask.detach()

		return loss, loss_s, loss_u, loss_sc, loss_uc

	def get_confidence_mask(self, pred_weak: Tensor, pred_count: Tensor, dim: int) -> Tensor:
		u_pred_sorted = pred_weak.sort(dim=dim, descending=True)[0]
		u_counts = pred_count.argmax(dim=dim)

		mean_values = [
			pred[:count].mean() if count > 0 else 0.0
			for pred, count in zip(u_pred_sorted, u_counts)
		]
		mean_values = torch.as_tensor(mean_values).cuda()

		mask = (mean_values > self.threshold_confidence).float()
		return mask

	def get_last_mask(self) -> Optional[Tensor]:
		return self.last_mask

	def get_lambda_u(self) -> float:
		return self.lambda_u
