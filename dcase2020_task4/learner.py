from abc import ABC
from time import time
from dcase2020_task4.trainer_abc import TrainerABC, SSTrainerABC
from dcase2020_task4.validator import ValidatorABC


class LearnerABC(ABC):
	def start(self):
		raise NotImplementedError("Abstract method")


class DefaultLearner(LearnerABC):
	def __init__(
			self,
			name: str,
			trainer: TrainerABC,
			validator: ValidatorABC,
			nb_epochs: int,
			scheduler=None,
			verbose: int = 1
	):
		self.name = name
		self.trainer = trainer
		self.validator = validator
		self.nb_epochs = nb_epochs
		self.scheduler = scheduler
		self.verbose = verbose

		self.start_time = None

	def start(self):
		self._on_start()

		for e in range(self.nb_epochs):
			self.trainer.train(e)
			self.validator.val(e)

			if self.scheduler is not None:
				self.scheduler.step()

		self._on_end()

	def _on_start(self):
		if self.verbose > 0:
			if issubclass(type(self.trainer), SSTrainerABC):
				print("\nStart %s training (%d epochs, %d supervised train examples, %d unsupervised train examples, "
					  "%d valid examples)..." % (
						  self.name,
						  self.nb_epochs,
						  self.trainer.nb_examples_supervised(),
						  self.trainer.nb_examples_unsupervised(),
						  self.validator.nb_examples()
					  ))
			else:
				print("\nStart %s training (%d epochs, %d train examples, %d valid examples)..." % (
					self.name,
					self.nb_epochs,
					self.trainer.nb_examples(),
					self.validator.nb_examples()
				))
		self.start_time = time()

	def _on_end(self):
		if self.verbose > 0:
			print("End %s training. (duration = %.2f)" % (self.name, time() - self.start_time))
