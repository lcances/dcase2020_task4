
from abc import ABC
from time import time
from dcase2020_task4.trainer import Trainer
from dcase2020_task4.validator import Validator


class Learner(ABC):
	def start(self):
		raise NotImplementedError("Abstract method")


class DefaultLearner(Learner):
	def __init__(
		self, name: str, trainer: Trainer, validator: Validator, nb_epochs: int, scheduler=None, verbose: int = 1
	):
		self.name = name
		self.trainer = trainer
		self.validator = validator
		self.nb_epochs = nb_epochs
		self.scheduler = scheduler
		self.verbose = verbose

	def start(self):
		if self.verbose > 0:
			print("\nStart %s training (%d epochs, %d train examples, %d valid examples)..." % (
				self.name,
				self.nb_epochs,
				self.trainer.nb_examples(),
				self.validator.nb_examples()
			))
		start = time()

		for e in range(self.nb_epochs):
			self.trainer.train(e)
			self.validator.val(e)

			if self.scheduler is not None:
				self.scheduler.step()

		if self.verbose > 0:
			print("End %s training. (duration = %.2f)" % (self.name, time() - start))
