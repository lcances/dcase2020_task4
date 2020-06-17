
from abc import ABC


class ValidatorABC(ABC):
	def val(self, epoch: int):
		raise NotImplementedError("Abstract method")

	def nb_examples(self) -> int:
		raise NotImplementedError("Abstract method")
