from abc import ABC

from torch import Tensor


class MixMatchLossABC(ABC):
	def __call__(self, s_pred: Tensor, s_target: Tensor, u_pred: Tensor, u_target: Tensor) -> (Tensor, Tensor, Tensor):
		raise NotImplementedError("Abstract method")
