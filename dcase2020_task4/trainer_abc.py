from abc import ABC


class TrainerABC(ABC):
	def train(self, epoch: int):
		raise NotImplementedError("Abstract method")

	def nb_examples(self) -> int:
		raise NotImplementedError("Abstract method")

	def reset_metrics(self):
		raise NotImplementedError("Abstract method")


class SSTrainerABC(TrainerABC, ABC):
	def nb_examples_supervised(self) -> int:
		raise NotImplementedError("Abstract method")

	def nb_examples_unsupervised(self) -> int:
		raise NotImplementedError("Abstract method")

	def nb_examples(self) -> int:
		return self.nb_examples_supervised() + self.nb_examples_unsupervised()
