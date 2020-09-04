
from argparse import Namespace
from torch import Tensor
from torch.nn.functional import binary_cross_entropy

from dcase2020_task4.mixmatch.losses.abc import MixMatchLossLocABC


class MixMatchLossMultiHotLoc(MixMatchLossLocABC):
	""" MixMatch loss component. """

	def __init__(self, lambda_u: float = 1.0):
		self.lambda_u = lambda_u

		self.criterion_s_weak = binary_cross_entropy
		self.criterion_u_weak = binary_cross_entropy
		self.criterion_s_strong = binary_cross_entropy
		self.criterion_u_strong = binary_cross_entropy

	@staticmethod
	def from_args(args: Namespace) -> 'MixMatchLossMultiHotLoc':
		return MixMatchLossMultiHotLoc(hparams.lambda_u)

	def __call__(
		self,
		s_pred_weak: Tensor, s_target_weak: Tensor, s_pred_strong: Tensor, s_target_strong: Tensor,
		u_pred_weak: Tensor, u_target_weak: Tensor, u_pred_strong: Tensor, u_target_strong: Tensor,
	) -> (Tensor, Tensor, Tensor, Tensor, Tensor):
		loss_s_weak = self.criterion_s_weak(s_pred_weak, s_target_weak)
		loss_s_weak = loss_s_weak.mean()

		loss_s_strong = self.criterion_s_strong(s_pred_strong, s_target_strong)
		loss_s_strong = loss_s_strong.mean()

		loss_u_weak = self.criterion_u_weak(u_pred_weak, u_target_weak)
		loss_u_weak = loss_u_weak.mean()

		loss_u_strong = self.criterion_u_strong(u_pred_strong, u_target_strong)
		loss_u_strong = loss_u_strong.mean()

		loss = loss_s_weak + loss_s_strong + self.lambda_u * (loss_u_weak + loss_u_strong)

		return loss, loss_s_weak, loss_u_weak, loss_s_strong, loss_u_strong

	def get_lambda_u(self) -> float:
		return self.lambda_u
