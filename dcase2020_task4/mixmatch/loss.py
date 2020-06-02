
from torch import Tensor
from torch.nn.functional import binary_cross_entropy
from typing import Callable

from dcase2020_task4.util.utils_match import cross_entropy


class MixMatchLoss(Callable):
	""" MixMatch loss component. """

	def __init__(
		self, lambda_u: float = 1.0, mode: str = "onehot", criterion_unsupervised: str = "sqdiff"
	):
		self.lambda_u = lambda_u
		self.mode = mode
		self.unsupervised_loss_mode = criterion_unsupervised

		if self.mode == "onehot":
			self.criterion_s = cross_entropy

			if criterion_unsupervised == "sqdiff":
				self.criterion_u = lambda pred_u, targets_u: (pred_u - targets_u) ** 2
			elif criterion_unsupervised == "crossentropy":
				self.criterion_u = cross_entropy
			else:
				raise RuntimeError("Invalid argument \"mode = %s\". Use %s." % (mode, " or ".join(("sqdiff", "crossentropy"))))

		elif self.mode == "multihot":
			self.criterion_s = binary_cross_entropy
			self.criterion_u = binary_cross_entropy

		else:
			raise RuntimeError("Invalid argument \"mode = %s\". Use %s." % (mode, " or ".join(("onehot", "multihot"))))

	def __call__(self, pred_s: Tensor, targets_x: Tensor, pred_u: Tensor, targets_u: Tensor) -> Tensor:
		loss_s = self.criterion_s(pred_s, targets_x)
		loss_s = loss_s.mean()

		loss_u = self.criterion_u(pred_u, targets_u)
		loss_u = loss_u.mean()

		return loss_s + self.lambda_u * loss_u
