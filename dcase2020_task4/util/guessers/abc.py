from abc import ABC
from torch import Tensor
from typing import Callable, Optional


class GuesserABC(ABC, Callable):
	"""
		Abstract class for all guessers.
	"""
	def __call__(self, x: Tensor, dim: int) -> Tensor:
		raise NotImplementedError("Abstract method")

	def guess_label(self, x: Tensor, dim: int) -> Tensor:
		return self.__call__(x, dim)


class GuesserModelABC(GuesserABC):
	"""
		Use model to make prediction for guessing label.
		Also store the last prediction computed.
	"""
	def __call__(self, x: Tensor, dim: int) -> Tensor:
		raise NotImplementedError("Abstract method")

	def get_last_pred(self) -> Optional[Tensor]:
		raise NotImplementedError("Abstract method")


class GuesserPredABC(GuesserABC):
	"""
		Guesser that use a model prediction to compute a artificial label.
	"""
	def __call__(self, x: Tensor, dim: int) -> Tensor:
		raise NotImplementedError("Abstract method")


class GuesserCompose(GuesserABC):
	"""
		Guesser that compose other guessers to compute a artificial label.
	"""
	def __init__(self, *args):
		self.guessers = list(args)

	def __call__(self, x: Tensor, dim: int) -> Tensor:
		for guesser in self.guessers:
			x = guesser(x, dim)
		return x
