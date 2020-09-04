
from torch.optim.optimizer import Optimizer
from typing import List
from dcase2020_task4.util.utils_match import set_lr, get_lr


class StepLRScheduler:
	"""
		Scheduler that divide learning rate at each limit.
	"""

	def __init__(self, optim: Optimizer, lr0: float, lr_decay_ratio: float, epoch_steps: List[int]):
		self.optim = optim
		self.lr0 = lr0
		self.lr_decay_ratio = lr_decay_ratio
		self.epoch_steps = epoch_steps

		self.epoch = 0

	def step(self):
		self.epoch += 1
		if self.epoch in self.epoch_steps:
			lr = self.get_optim_lr()
			set_lr(self.optim, lr * self.lr_decay_ratio)

	def reset(self):
		self.epoch = 0
		set_lr(self.optim, self.lr0)

	def get_optim_lr(self) -> float:
		return get_lr(self.optim)
