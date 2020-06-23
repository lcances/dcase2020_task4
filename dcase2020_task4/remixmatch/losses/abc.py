
from abc import ABC
from torch import Tensor
from typing import Callable


class ReMixMatchLossABC(ABC, Callable):
	def __call__(
		self,
		pred_s: Tensor, targets_x: Tensor,
		pred_u: Tensor, targets_u: Tensor,
		pred_u1: Tensor, targets_u1: Tensor,
		pred_r: Tensor, targets_r: Tensor,
	) -> (Tensor, Tensor, Tensor, Tensor, Tensor):
		raise NotImplementedError("Abstract method")
