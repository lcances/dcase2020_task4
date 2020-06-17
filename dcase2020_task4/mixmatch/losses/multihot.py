
from torch import Tensor
from torch.nn.functional import binary_cross_entropy

from dcase2020_task4.mixmatch.losses.abc import MixMatchLossABC


class MixMatchLossMultiHot(MixMatchLossABC):
	""" MixMatch loss component. """

	def __init__(self, lambda_u: float = 1.0):
		self.lambda_u = lambda_u

		self.criterion_s = binary_cross_entropy
		self.criterion_u = binary_cross_entropy

	@staticmethod
	def from_edict(hparams) -> 'MixMatchLossMultiHot':
		return MixMatchLossMultiHot(hparams.lambda_u)

	def __call__(self, s_pred: Tensor, s_target: Tensor, u_pred: Tensor, u_target: Tensor) -> (Tensor, Tensor, Tensor):
		loss_s = self.criterion_s(s_pred, s_target)
		loss_s = loss_s.mean()

		loss_u = self.criterion_u(u_pred, u_target)
		loss_u = loss_u.mean()

		loss = loss_s + self.lambda_u * loss_u

		return loss, loss_s, loss_u
