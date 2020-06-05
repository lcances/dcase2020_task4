import numpy as np
import torch

from time import time
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable

from dcase2020.pytorch_metrics.metrics import Metrics

from dcase2020_task4.mixmatch.rampup import RampUp
from dcase2020_task4.trainer import SSTrainer
from dcase2020_task4.util.ZipLongestCycle import ZipLongestCycle
from dcase2020_task4.util.utils_match import get_lr


class MixMatchTrainer(SSTrainer):
	def __init__(
		self,
		model: Module,
		acti_fn: Callable,
		optim: Optimizer,
		loader_train_s_augm: DataLoader,
		loader_train_u_augms: DataLoader,
		metric_s: Metrics,
		metric_u: Metrics,
		writer: SummaryWriter,
		criterion: Callable,
		mixer: Callable,
		lambda_u_rampup: RampUp
	):
		self.model = model
		self.acti_fn = acti_fn
		self.optim = optim
		self.loader_train_s_augm = loader_train_s_augm
		self.loader_train_u_augms = loader_train_u_augms
		self.metric_s = metric_s
		self.metric_u = metric_u
		self.writer = writer
		self.criterion = criterion
		self.mixer = mixer
		self.lambda_u_rampup = lambda_u_rampup

	def train(self, epoch: int):
		train_start = time()
		self.metric_s.reset()
		self.metric_u.reset()
		self.model.train()

		losses, acc_train_s, acc_train_u = [], [], []
		zip_cycle = ZipLongestCycle([self.loader_train_s_augm, self.loader_train_u_augms])

		for i, ((batch_s_augm, labels_s), batch_u_augms) in enumerate(zip_cycle):
			batch_s_augm = batch_s_augm.cuda().float()
			labels_s = labels_s.cuda().float()
			batch_u_augms = torch.stack(batch_u_augms).cuda().float()

			# Apply mix
			batch_s_mixed, labels_s_mixed, batch_u_mixed, labels_u_mixed = self.mixer(batch_s_augm, labels_s, batch_u_augms)

			# Compute logits
			logits_s = self.model(batch_s_mixed)
			logits_u = self.model(batch_u_mixed)

			# Compute accuracies
			pred_s = self.acti_fn(logits_s, dim=1)
			pred_u = self.acti_fn(logits_u, dim=1)

			mean_acc_s = self.metric_s(pred_s, labels_s_mixed)
			mean_acc_u = self.metric_u(pred_u, labels_u_mixed)

			# Update model
			self.criterion.lambda_u = self.lambda_u_rampup.value()
			loss = self.criterion(pred_s, labels_s_mixed, pred_u, labels_u_mixed)
			self.optim.zero_grad()
			loss.backward()
			self.optim.step()

			self.lambda_u_rampup.step()

			# Store data
			losses.append(loss.item())
			acc_train_s.append(self.metric_s.value.item())
			acc_train_u.append(self.metric_u.value.item())

			print("Epoch {}, {:d}% \t loss: {:.4e} - acc_s: {:.4e} - acc_u: {:.4e} - took {:.2f}s".format(
				epoch + 1,
				int(100 * (i + 1) / len(zip_cycle)),
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
		return len(self.loader_train_s_augm) * self.loader_train_s_augm.batch_size

	def nb_examples_unsupervised(self) -> int:
		return len(self.loader_train_u_augms) * self.loader_train_u_augms.batch_size
