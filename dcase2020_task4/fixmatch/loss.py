from torch import Tensor
from torch.nn import BCELoss
from torch.nn.functional import binary_cross_entropy
from typing import Callable

from dcase2020_task4.util.utils_match import cross_entropy


class FixMatchLoss(Callable):
	def __init__(
		self,
		lambda_u: float = 1.0,
		threshold_mask: float = 0.95,
		mode: str = "onehot",
	):
		self.lambda_u = lambda_u
		self.threshold_mask = threshold_mask
		self.mode = mode

		if self.mode == "onehot":
			self.criterion_s = cross_entropy
			self.criterion_u = cross_entropy
		elif self.mode == "multihot":
			self.criterion_s = binary_cross_entropy
			# Note : use BCELoss instead of binary_cross_entropy because we need a loss per batch and not a mean reduction
			self.criterion_u = BCELoss(reduction="none")
		else:
			raise RuntimeError("Invalid argument \"mode = %s\". Use %s." % (self.mode, " or ".join(("onehot", "multihot"))))

	@staticmethod
	def from_edict(hparams) -> 'FixMatchLoss':
		return FixMatchLoss(hparams.lambda_u, hparams.threshold_mask, hparams.mode)

	def __call__(
		self, pred_s_weak: Tensor, labels_s: Tensor, pred_u_weak: Tensor, pred_u_strong: Tensor, labels_u_guessed: Tensor,
	) -> Tensor:
		if pred_s_weak.size() != labels_s.size():
			raise RuntimeError("Weak predictions and labels must have the same size.")
		if pred_u_weak.size() != pred_u_strong.size():
			raise RuntimeError("Weak predictions and strong predictions must have the same size.")

		# Supervised loss
		loss_s = self.criterion_s(pred_s_weak, labels_s)
		loss_s = loss_s.mean()

		# Unsupervised loss
		max_values, _ = pred_u_weak.max(dim=1)
		mask = (max_values > self.threshold_mask).float()
		loss_u = self.criterion_u(pred_u_strong, labels_u_guessed)
		if self.mode == "multihot":
			loss_u = loss_u.mean(dim=1)
		loss_u *= mask
		loss_u = loss_u.mean()

		loss = loss_s + self.lambda_u * loss_u

		return loss


class FixMatchLossWS(Callable):
	def __init__(self,
		lambda_u: float = 1.0,
		threshold_mask: float = 0.5):
		self.lambda_u = lambda_u
		self.threshold_mask = threshold_mask

		self.criterion = BCELoss(reduction="none")

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
		loss_s_at = self.criterion(pred_s_weak_at, labels_s_at).mean()
		loss_s_loc = self.criterion(pred_s_weak_loc, labels_s_loc).mean()

		loss_u_at = self.criterion(pred_u_strong_at, labels_u_guessed_at)
		loss_u_loc = self.criterion(pred_u_strong_loc, labels_u_guessed_loc)
		raise NotImplementedError("TODO")

		return loss_s_at + loss_s_loc
