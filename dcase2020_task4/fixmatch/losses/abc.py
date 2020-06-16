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
	) -> (Tensor, Tensor, Tensor):
		raise NotImplementedError("Abstract method")


class FixMatchLossMultiHotLocABC(ABC, Callable):
	def __call__(
		self,
		s_pred_weak_augm_weak: Tensor, s_labels_weak: Tensor,
		u_pred_weak_augm_weak: Tensor, u_pred_weak_augm_strong: Tensor, u_labels_weak_guessed: Tensor,
		s_pred_strong_augm_weak: Tensor, s_labels_strong: Tensor,
		u_pred_strong_augm_weak: Tensor, u_pred_strong_augm_strong: Tensor, u_labels_strong_guessed: Tensor,
	) -> (Tensor, Tensor, Tensor, Tensor, Tensor):
		raise NotImplementedError("Abstract method")
