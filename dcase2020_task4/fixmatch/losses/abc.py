from abc import ABC
from torch import Tensor
from typing import Callable


class FixMatchLossABC(ABC, Callable):
	def __call__(
		self,
		s_pred_weak_augm_weak: Tensor,
		s_labels_weak: Tensor,
		u_pred_weak_augm_weak: Tensor,
		u_pred_weak_augm_strong: Tensor,
		u_labels_weak_guessed: Tensor,
	) -> Tensor:
		raise NotImplementedError("Abstract method")
