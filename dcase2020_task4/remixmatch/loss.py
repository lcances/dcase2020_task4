from torch import Tensor
from torch.nn.functional import binary_cross_entropy_with_logits
from typing import Callable

from ..util.utils_match import cross_entropy_with_logits


class ReMixMatchLoss(Callable):
	def __init__(
		self, acti_fn: Callable, lambda_u: float = 1.5, lambda_u1: float = 0.5, lambda_r: float = 0.5, mode: str = "onehot"
	):
		self.acti_fn = acti_fn
		self.lambda_u = lambda_u
		self.lambda_u1 = lambda_u1
		self.lambda_r = lambda_r
		self.mode = mode

		if self.mode == "onehot":
			self.criterion = cross_entropy_with_logits
		elif self.mode == "multihot":
			self.criterion = binary_cross_entropy_with_logits
		else:
			raise RuntimeError("Invalid argument \"mode = %s\". Use %s." % (mode, " or ".join(("onehot", "multihot"))))

	def __call__(
		self,
		logits_x: Tensor, targets_x: Tensor,
		logits_u: Tensor, targets_u: Tensor,
		logits_u1: Tensor, targets_u1: Tensor,
		logits_r: Tensor, targets_r: Tensor,
	) -> Tensor:
		loss_x = self.criterion(logits_x, targets_x).mean()
		loss_u = self.criterion(logits_u, targets_u).mean()
		loss_u1 = self.criterion(logits_u1, targets_u1).mean()
		loss_r = self.criterion(logits_r, targets_r).mean()

		loss = loss_x + self.lambda_u * loss_u + self.lambda_u1 * loss_u1 + self.lambda_r * loss_r

		return loss
