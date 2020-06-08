import numpy as np

from time import time
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable

from metric_utils.metrics import Metrics

from dcase2020_task4.util.zip_cycle import ZipCycle
from dcase2020_task4.util.utils_match import binarize_onehot_labels, get_lr
from dcase2020_task4.trainer import SSTrainer


class FixMatchTrainer(SSTrainer):
	def __init__(
		self,
		model: Module,
		acti_fn: Callable,
		optim: Optimizer,
		loader_train_s_weak: DataLoader,
		loader_train_u_weak_strong: DataLoader,
		metric_s_at: Metrics,
		metric_s_loc: Metrics,
		metric_u_at: Metrics,
		metric_u_loc: Metrics,
		writer: SummaryWriter,
		criterion: Callable,
		threshold_multihot: float = 0.5,
	):
		self.model = model
		self.acti_fn = acti_fn
		self.optim = optim
		self.loader_train_s_weak = loader_train_s_weak
		self.loader_train_u_weak_strong = loader_train_u_weak_strong
		self.metric_s_at = metric_s_at
		self.metric_s_loc = metric_s_loc
		self.metric_u_at = metric_u_at
		self.metric_u_loc = metric_u_loc
		self.writer = writer
		self.criterion = criterion
		self.threshold_multihot = threshold_multihot

		self.mode = "multihot"

	def train(self, epoch: int):
		train_start = time()
		self.metric_s_at.reset()
		self.metric_s_loc.reset()
		self.metric_u_at.reset()
		self.metric_u_loc.reset()
		self.model.train()

		losses, acc_train_s_at, acc_train_s_loc, acc_train_u_at, acc_train_u_loc = [], [], [], [], []
		zip_cycle = ZipCycle([self.loader_train_s_weak, self.loader_train_u_weak_strong])
		iter_train = iter(zip_cycle)

		for i, item in enumerate(iter_train):
			(batch_s_weak, labels_s_at, labels_s_loc), (batch_u_weak, batch_u_strong) = item

			batch_s_weak = batch_s_weak.cuda().float()
			labels_s_at = labels_s_at.cuda().float()
			labels_s_loc = labels_s_loc.cuda().float()
			batch_u_weak = batch_u_weak.cuda().float()
			batch_u_strong = batch_u_strong.cuda().float()

			# Compute logits
			logits_s_weak_at, logits_s_weak_loc = self.model(batch_s_weak)
			logits_u_weak_at, logits_u_weak_loc = self.model(batch_u_weak)
			logits_u_strong_at, logits_u_strong_loc = self.model(batch_u_strong)

			# Compute accuracies
			pred_s_weak_at = self.acti_fn(logits_s_weak_at, dim=1)
			pred_u_weak_at = self.acti_fn(logits_u_weak_at, dim=1)
			pred_u_strong_at = self.acti_fn(logits_u_strong_at, dim=1)
			pred_s_weak_loc = self.acti_fn(logits_s_weak_loc, dim=1)
			pred_u_weak_loc = self.acti_fn(logits_u_weak_loc, dim=1)
			pred_u_strong_loc = self.acti_fn(logits_u_strong_loc, dim=1)

			labels_u_guessed_at = (pred_u_weak_at > self.threshold_multihot).float()
			labels_u_guessed_loc = (pred_u_weak_loc > self.threshold_multihot).float()

			# Update model
			loss = self.criterion(
				pred_s_weak_at,
				labels_s_at,
				pred_s_weak_loc,
				labels_s_loc,
				pred_u_weak_at,
				pred_u_strong_at,
				labels_u_guessed_at,
				pred_u_weak_loc,
				pred_u_strong_loc,
				labels_u_guessed_loc,
			)
			self.optim.zero_grad()
			loss.backward()
			self.optim.step()

			# Store data
			losses.append(loss.item())
			acc_train_s_at.append(self.metric_s_at.value.item())
			acc_train_u_at.append(self.metric_u_at.value.item())
			acc_train_s_loc.append(self.metric_s_loc.value.item())
			acc_train_u_loc.append(self.metric_u_loc.value.item())

			mean_acc_s_at = self.metric_s_at(pred_s_weak_at, labels_s_at)
			mean_acc_u_at = self.metric_u_at(pred_u_strong_at, labels_u_guessed_at)
			mean_acc_s_loc = self.metric_s_loc(pred_s_weak_loc, labels_s_loc)
			mean_acc_u_loc = self.metric_u_loc(pred_u_strong_loc, labels_u_guessed_loc)

			values = {
				"loss": loss.item(),
				"acc_s_at": mean_acc_s_at,
				"acc_u_at": mean_acc_u_at,
				"acc_s_loc": mean_acc_s_loc,
				"acc_u_loc": mean_acc_u_loc,
			}
			print("Epoch {}, {:d}% \t {:s} - took {:.2f}s".format(
				epoch + 1,
				int(100 * (i + 1) / len(zip_cycle)),
				" - ".join(["{:s}: {:.4e}".format(name, value) for name, value in values.items()]),
				time() - train_start
			), end="\r")

		print("")

		self.writer.add_scalar("train/loss", float(np.mean(losses)), epoch)
		self.writer.add_scalar("train/acc_s_at", float(np.mean(acc_train_s_at)), epoch)
		self.writer.add_scalar("train/acc_u_at", float(np.mean(acc_train_u_at)), epoch)
		self.writer.add_scalar("train/acc_s_loc", float(np.mean(acc_train_s_loc)), epoch)
		self.writer.add_scalar("train/acc_u_loc", float(np.mean(acc_train_u_loc)), epoch)
		self.writer.add_scalar("train/lr", get_lr(self.optim), epoch)

	def nb_examples_supervised(self) -> int:
		return len(self.loader_train_s_weak) * self.loader_train_s_weak.batch_size

	def nb_examples_unsupervised(self) -> int:
		return len(self.loader_train_u_weak_strong) * self.loader_train_u_weak_strong.batch_size
