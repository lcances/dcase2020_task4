from abc import ABC

from torch import Tensor
from typing import Callable


class MixMatchLossTagABC(ABC, Callable):
	def __call__(self, s_pred: Tensor, s_target: Tensor, u_pred: Tensor, u_target: Tensor) -> (Tensor, Tensor, Tensor):
		raise NotImplementedError("Abstract method")

	def get_lambda_s(self) -> float:
		raise NotImplementedError("Abstract method")

	def get_lambda_u(self) -> float:
		raise NotImplementedError("Abstract method")


class MixMatchLossLocABC(ABC, Callable):
	def __call__(
		self,
		s_pred_weak: Tensor, s_target_weak: Tensor, s_pred_strong: Tensor, s_target_strong: Tensor,
		u_pred_weak: Tensor, u_target_weak: Tensor, u_pred_strong: Tensor, u_target_strong: Tensor,
	) -> (Tensor, Tensor, Tensor, Tensor, Tensor):
		raise NotImplementedError("Abstract method")

	def get_lambda_u(self) -> float:
		raise NotImplementedError("Abstract method")
