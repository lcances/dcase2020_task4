import torch

from torch import Tensor
from torch.nn.functional import binary_cross_entropy
from typing import Callable

from ..util.utils_match import cross_entropy


class MixMatchLoss(Callable):
	def __init__(
		self, lambda_u: float = 1.0, mode: str = "onehot", criterion_unsupervised: str = "l2norm"
	):
		self.lambda_u = lambda_u
		self.mode = mode
		self.unsupervised_loss_mode = criterion_unsupervised

		if self.mode == "onehot":
			self.criterion_s = cross_entropy

			if criterion_unsupervised == "l2norm":
				self.criterion_u = lambda pred_u, targets_u: torch.mean((pred_u - targets_u) ** 2)
			elif criterion_unsupervised == "crossentropy":
				self.criterion_u = cross_entropy
			else:
				raise RuntimeError("Invalid argument \"mode = %s\". Use %s." % (mode, " or ".join(("l2norm", "crossentropy"))))

		elif self.mode == "multihot":
			self.criterion_s = binary_cross_entropy
			self.criterion_u = binary_cross_entropy

		else:
			raise RuntimeError("Invalid argument \"mode = %s\". Use %s." % (mode, " or ".join(("onehot", "multihot"))))

	def __call__(self, pred_x: Tensor, targets_x: Tensor, pred_u: Tensor, targets_u: Tensor) -> Tensor:
		loss_x = self.criterion_s(pred_x, targets_x).mean()
		loss_x = loss_x.mean()

		loss_u = self.criterion_u(pred_u, targets_u)
		loss_u = loss_u.mean()

		return loss_x + self.lambda_u * loss_u
