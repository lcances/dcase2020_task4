
from abc import ABC
from time import time
from typing import Optional

from dcase2020_task4.trainer_abc import TrainerABC, TrainerABC
from dcase2020_task4.validator import ValidatorABC


class LearnerABC(ABC):
	def start(self):
		raise NotImplementedError("Abstract method")


class Learner(LearnerABC):
	"""
		Class used to
	"""
	def __init__(
		self,
		name: str,
		trainer: TrainerABC,
		validator: ValidatorABC,
		nb_epochs: int,
		steppables: Optional[list] = None,
		verbose: int = 1
	):
		self.name = name
		self.trainer = trainer
		self.validator = validator
		self.nb_epochs = nb_epochs
		self.steppables = steppables if steppables is not None else []
		self.verbose = verbose

		self.start_time = None

	def start(self):
		self._on_start()

		for e in range(self.nb_epochs):
			self.trainer.train(e)
			self.validator.val(e)

			for steppable in self.steppables:
				steppable.step()

		self._on_end()

	def _on_start(self):
		if self.verbose > 0:
			print("\nStart %s training (%d epochs)..." % (self.name, self.nb_epochs))
		self.start_time = time()

	def _on_end(self):
		if self.verbose > 0:
			print("End %s training. (duration = %.2f)" % (self.name, time() - self.start_time))
