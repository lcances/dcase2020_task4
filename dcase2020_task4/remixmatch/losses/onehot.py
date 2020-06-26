from torch import Tensor

from dcase2020_task4.remixmatch.losses.abc import ReMixMatchLossTagABC
from dcase2020_task4.util.utils_match import cross_entropy


class ReMixMatchLossOneHot(ReMixMatchLossTagABC):
	def __init__(
		self, lambda_u: float = 1.5, lambda_u1: float = 0.5, lambda_r: float = 0.5
	):
		self.lambda_u = lambda_u
		self.lambda_u1 = lambda_u1
		self.lambda_r = lambda_r

		self.criterion_s = cross_entropy
		self.criterion_u = cross_entropy
		self.criterion_u1 = cross_entropy
		self.criterion_r = cross_entropy

	@staticmethod
	def from_edict(hparams) -> 'ReMixMatchLossOneHot':
		return ReMixMatchLossOneHot(hparams.lambda_u, hparams.lambda_u1, hparams.lambda_r)

	def __call__(
		self,
		pred_s: Tensor, targets_x: Tensor,
		pred_u: Tensor, targets_u: Tensor,
		pred_u1: Tensor, targets_u1: Tensor,
		pred_r: Tensor, targets_r: Tensor,
	) -> (Tensor, Tensor, Tensor, Tensor, Tensor):
		loss_s = self.criterion_s(pred_s, targets_x)
		loss_s = loss_s.mean()

		loss_u = self.criterion_u(pred_u, targets_u)
		loss_u = loss_u.mean()

		loss_u1 = self.criterion_u1(pred_u1, targets_u1)
		loss_u1 = loss_u1.mean()

		loss_r = self.criterion_r(pred_r, targets_r)
		loss_r = loss_r.mean()

		loss = loss_s + self.lambda_u * loss_u + self.lambda_u1 * loss_u1 + self.lambda_r * loss_r

		return loss, loss_s, loss_u, loss_u1, loss_r
