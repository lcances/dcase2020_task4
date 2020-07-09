
from torch import Tensor

from dcase2020_task4.mixmatch.losses.abc import MixMatchLossTagABC
from dcase2020_task4.util.utils_match import cross_entropy


class MixMatchLossOneHot(MixMatchLossTagABC):
	""" MixMatch loss component. """

	def __init__(
		self,
		lambda_u: float = 1.0,
		criterion_name_u: str = "sq_diff"
	):
		self.lambda_u = lambda_u
		self.unsupervised_loss_mode = criterion_name_u

		self.criterion_s = cross_entropy

		if criterion_name_u == "sq_diff":
			self.criterion_u = sq_diff
		elif criterion_name_u == "cross_entropy":
			self.criterion_u = cross_entropy
		else:
			raise RuntimeError("Invalid argument \"mode = %s\". Use %s." % (criterion_name_u, " or ".join(("sq_diff", "cross_entropy"))))

	@staticmethod
	def from_edict(hparams) -> 'MixMatchLossOneHot':
		return MixMatchLossOneHot(hparams.lambda_u, hparams.criterion_name_u)

	def __call__(self, s_pred: Tensor, s_target: Tensor, u_pred: Tensor, u_target: Tensor) -> (Tensor, Tensor, Tensor):
		loss_s = self.criterion_s(s_pred, s_target)
		loss_s = loss_s.mean()

		loss_u = self.criterion_u(u_pred, u_target)
		loss_u = loss_u.mean()

		loss = loss_s + self.lambda_u * loss_u

		return loss, loss_s, loss_u

	def get_lambda_u(self) -> float:
		return self.lambda_u


def sq_diff(a: Tensor, b: Tensor) -> Tensor:
	return (a - b) ** 2
