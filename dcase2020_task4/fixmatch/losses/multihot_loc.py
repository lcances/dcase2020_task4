from torch import Tensor
from torch.nn import BCELoss
from typing import Callable


class FixMatchLossMultiHotLoc(Callable):
	def __init__(self,
		lambda_u: float = 1.0,
		threshold_mask: float = 0.5
	):
		self.lambda_u = lambda_u
		self.threshold_mask = threshold_mask

		self.criterion = BCELoss(reduction="none")
		raise NotImplementedError("TODO")

	def __call__(
		self,
		pred_s_weak_at: Tensor,
		labels_s_at: Tensor,
		pred_s_weak_loc: Tensor,
		labels_s_loc: Tensor,
		pred_u_weak_at: Tensor,
		pred_u_strong_at: Tensor,
		labels_u_guessed_at: Tensor,
		pred_u_weak_loc: Tensor,
		pred_u_strong_loc: Tensor,
		labels_u_guessed_loc: Tensor,
	) -> Tensor:
		"""
		loss_s_at = self.criterion(pred_s_weak_at, labels_s_at).mean()
		loss_s_loc = self.criterion(pred_s_weak_loc, labels_s_loc).mean()

		loss_u_at = self.criterion(pred_u_strong_at, labels_u_guessed_at)
		loss_u_loc = self.criterion(pred_u_strong_loc, labels_u_guessed_loc)

		return loss_s_at + loss_s_loc
		"""
		raise NotImplementedError("TODO")
