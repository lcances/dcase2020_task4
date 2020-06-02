import numpy as np

from easydict import EasyDict as edict
from time import time
from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable

from dcase2020.pytorch_metrics.metrics import Metrics
from dcase2020_task4.util.utils_match import get_lr
from dcase2020_task4.trainer import Trainer


class SupervisedTrainer(Trainer):
	def __init__(
		self,
		model: Module,
		acti_fn: Callable,
		optim: Optimizer,
		loader: DataLoader,
		criterion: Callable,
		metrics: Metrics,
		writer: SummaryWriter,
	):
		self.model = model
		self.acti_fn = acti_fn
		self.optim = optim
		self.loader = loader
		self.criterion = criterion
		self.metrics = metrics
		self.writer = writer

	def train(self, epoch: int):
		train_start = time()
		self.metrics.reset()
		self.model.train()

		losses, acc_train = [], []
		iter_train = iter(self.loader)

		for i, (x, y) in enumerate(iter_train):
			x = x.cuda().float()
			y = y.cuda().float()

			# Compute logits
			logits = self.model(x)
			pred = self.acti_fn(logits, dim=1)

			# Compute accuracy
			accuracy = self.metrics(pred, y)

			# Update model
			loss = self.criterion(logits, y).mean()
			self.optim.zero_grad()
			loss.backward()
			self.optim.step()

			# Store data
			losses.append(loss.item())
			acc_train.append(self.metrics.value.item())

			# logs
			print("Epoch {}, {:d}% \t loss: {:.4e} - acc: {:.4e} - took {:.2f}s".format(
				epoch + 1,
				int(100 * (i + 1) / len(self.loader)),
				loss.item(),
				accuracy,
				time() - train_start
			), end="\r")

		print("")

		self.writer.add_scalar("train/loss", float(np.mean(losses)), epoch)
		self.writer.add_scalar("train/acc", float(np.mean(acc_train)), epoch)
		self.writer.add_scalar("train/lr", get_lr(self.optim), epoch)

	def nb_examples(self) -> int:
		return len(self.loader) * self.loader.batch_size
