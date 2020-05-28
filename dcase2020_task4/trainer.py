from abc import ABC


class Trainer(ABC):
	def train(self, epoch: int):
		raise NotImplementedError("Abstract method")

	def nb_examples(self) -> int:
		raise NotImplementedError("Abstract method")


class SSTrainer(Trainer, ABC):
	def nb_examples_supervised(self) -> int:
		raise NotImplementedError("Abstract method")

	def nb_examples_unsupervised(self) -> int:
		raise NotImplementedError("Abstract method")

	def nb_examples(self) -> int:
		return self.nb_examples_supervised() + self.nb_examples_unsupervised()
