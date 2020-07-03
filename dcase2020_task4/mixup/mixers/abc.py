
from abc import ABC
from torch import Tensor
from typing import Callable


class MixUpMixerTagABC(ABC, Callable):
	def __call__(self, batch_1: Tensor, labels_1: Tensor, batch_2: Tensor, labels_2: Tensor) -> (Tensor, Tensor):
		raise NotImplementedError("Abstract method")
