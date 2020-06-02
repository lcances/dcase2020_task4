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

from dcase2020_task4.mixmatch.rampup import RampUp
from dcase2020_task4.trainer import SSTrainer
from dcase2020_task4.util.MergeDataLoader import MergeDataLoader
from dcase2020_task4.util.utils_match import get_lr


class MixMatchTrainer(SSTrainer):
	def __init__(
		self,
		model: Module,
		acti_fn: Callable,
		optim: Optimizer,
		loader_train_s: DataLoader,
		loader_train_u: DataLoader,
		augm_fn: Callable,
		metrics_s: Metrics,
		metrics_u: Metrics,
		writer: SummaryWriter,
		criterion: Callable,
		mixer: Callable,
		lambda_u_rampup: RampUp
	):
		self.model = model
		self.acti_fn = acti_fn
		self.optim = optim
		self.loader_train_s = loader_train_s
		self.loader_train_u = loader_train_u
		self.augm_fn = augm_fn
		self.metrics_s = metrics_s
		self.metrics_u = metrics_u
		self.writer = writer
		self.criterion = criterion
		self.mixer = mixer
		self.lambda_u_rampup = lambda_u_rampup

	def train(self, epoch: int):
		train_start = time()
		self.metrics_s.reset()
		self.metrics_u.reset()
		self.model.train()

		losses, acc_train_s, acc_train_u = [], [], []
		loader_merged = MergeDataLoader([self.loader_train_s, self.loader_train_u])
		iter_train = iter(loader_merged)

		for i, (batch_s, labels_s, batch_u) in enumerate(iter_train):
			batch_s = batch_s.cuda().float()
			labels_s = labels_s.cuda().float()
			batch_u = batch_u.cuda().float()

			# Apply mix
			batch_s_mixed, labels_s_mixed, batch_u_mixed, labels_u_mixed = self.mixer(batch_s, labels_s, batch_u)

			# Compute logits
			logits_s = self.model(batch_s_mixed)
			logits_u = self.model(batch_u_mixed)

			# Compute accuracies
			pred_s = self.acti_fn(logits_s)
			pred_u = self.acti_fn(logits_u)

			mean_acc_s = self.metrics_s(pred_s, labels_s_mixed)
			mean_acc_u = self.metrics_u(pred_u, labels_u_mixed)

			# Update model
			self.criterion.lambda_u = self.lambda_u_rampup.value()
			loss = self.criterion(pred_s, labels_s_mixed, pred_u, labels_u_mixed)
			self.optim.zero_grad()
			loss.backward()
			self.optim.step()

			self.lambda_u_rampup.step()

			# Store data
			losses.append(loss.item())
			acc_train_s.append(self.metrics_s.value.item())
			acc_train_u.append(self.metrics_u.value.item())

			print("Epoch {}, {:d}% \t loss: {:.4e} - acc_s: {:.4e} - acc_u: {:.4e} - took {:.2f}s".format(
				epoch + 1,
				int(100 * (i + 1) / len(loader_merged)),
				loss.item(),
				mean_acc_s,
				mean_acc_u,
				time() - train_start
			), end="\r")

		print("")

		self.writer.add_scalar("train/loss", float(np.mean(losses)), epoch)
		self.writer.add_scalar("train/acc_s", float(np.mean(acc_train_s)), epoch)
		self.writer.add_scalar("train/acc_u", float(np.mean(acc_train_u)), epoch)
		self.writer.add_scalar("train/lr", get_lr(self.optim), epoch)

	def nb_examples_supervised(self) -> int:
		return len(self.loader_train_s) * self.loader_train_s.batch_size

	def nb_examples_unsupervised(self) -> int:
		return len(self.loader_train_u) * self.loader_train_u.batch_size
