
from abc import ABC
from torch import Tensor
from typing import Callable


class ReMixMatchLossTagABC(ABC, Callable):
	def __call__(
		self,
		pred_s: Tensor, targets_x: Tensor,
		pred_u: Tensor, targets_u: Tensor,
		pred_u1: Tensor, targets_u1: Tensor,
		pred_r: Tensor, targets_r: Tensor,
	) -> (Tensor, Tensor, Tensor, Tensor, Tensor):
		raise NotImplementedError("Abstract method")

	def get_lambda_s(self) -> float:
		raise NotImplementedError("Abstract method")

	def get_lambda_u(self) -> float:
		raise NotImplementedError("Abstract method")

	def get_lambda_u1(self) -> float:
		raise NotImplementedError("Abstract method")

	def get_lambda_r(self) -> float:
		raise NotImplementedError("Abstract method")
