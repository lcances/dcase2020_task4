import numpy as np

from torch.optim.optimizer import Optimizer
from dcase2020_task4.util.utils_match import set_lr


class CosineLRScheduler:
	def __init__(self, optim: Optimizer, nb_epochs: int, lr0: float = 3e-3):
		self.optim = optim
		self.nb_epochs = nb_epochs
		self.lr0 = lr0
		self.epoch = 0

	def step(self):
		new_lr = self.lr0 * np.cos(7.0 * np.pi * self.epoch / (16.0 * self.nb_epochs))
		set_lr(self.optim, new_lr)

	def reset(self):
		self.epoch = 0
