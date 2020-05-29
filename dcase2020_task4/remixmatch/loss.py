from torch import Tensor
from torch.nn.functional import binary_cross_entropy
from typing import Callable

from ..util.utils_match import cross_entropy


class ReMixMatchLoss(Callable):
	def __init__(
		self, lambda_u: float = 1.5, lambda_u1: float = 0.5, lambda_r: float = 0.5, mode: str = "onehot"
	):
		self.lambda_u = lambda_u
		self.lambda_u1 = lambda_u1
		self.lambda_r = lambda_r
		self.mode = mode

		self.criterion_r = cross_entropy

		if self.mode == "onehot":
			self.criterion = cross_entropy
		elif self.mode == "multihot":
			self.criterion = binary_cross_entropy
		else:
			raise RuntimeError("Invalid argument \"mode = %s\". Use %s." % (mode, " or ".join(("onehot", "multihot"))))

	def __call__(
		self,
		pred_x: Tensor, targets_x: Tensor,
		pred_u: Tensor, targets_u: Tensor,
		pred_u1: Tensor, targets_u1: Tensor,
		pred_r: Tensor, targets_r: Tensor,
	) -> Tensor:
		loss_x = self.criterion(pred_x, targets_x)
		loss_u = self.criterion(pred_u, targets_u)
		loss_u1 = self.criterion(pred_u1, targets_u1)
		loss_r = self.criterion_r(pred_r, targets_r)

		loss_x = loss_x.mean()
		loss_u = loss_u.mean()
		loss_u1 = loss_u1.mean()
		loss_r = loss_r.mean()

		loss = loss_x + self.lambda_u * loss_u + self.lambda_u1 * loss_u1 + self.lambda_r * loss_r

		return loss
