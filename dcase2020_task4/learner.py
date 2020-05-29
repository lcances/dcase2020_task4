
from abc import ABC
from dcase2020_task4.trainer import Trainer
from dcase2020_task4.validator import Validator


class Learner(ABC):
	def start(self):
		raise NotImplementedError("Abstract method")


class DefaultLearner(Learner):
	def __init__(self, trainer: Trainer, validator: Validator, nb_epochs: int, scheduler=None):
		self.trainer = trainer
		self.validator = validator
		self.nb_epochs = nb_epochs
		self.scheduler = scheduler

	def start(self):
		for e in range(self.nb_epochs):
			self.trainer.train(e)
			self.validator.val(e)

			if self.scheduler is not None:
				self.scheduler.step()
