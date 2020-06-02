import numpy as np

from time import time
from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable

from dcase2020.pytorch_metrics.metrics import Metrics

from dcase2020_task4.util.MergeDataLoader import MergeDataLoader
from dcase2020_task4.util.utils_match import binarize_onehot_labels, get_lr
from dcase2020_task4.trainer import SSTrainer


class FixMatchTrainer(SSTrainer):
	def __init__(
		self,
		model: Module,
		acti_fn: Callable,
		optim: Optimizer,
		loader_train_s: DataLoader,
		loader_train_u: DataLoader,
		weak_augm_fn: Callable[[Tensor], Tensor],
		strong_augm_fn: Callable[[Tensor], Tensor],
		metrics_s: Metrics,
		metrics_u: Metrics,
		writer: SummaryWriter,
		criterion: Callable,
		mode: str = "onehot",
		threshold_multihot: float = 0.5,
	):
		self.model = model
		self.acti_fn = acti_fn
		self.optim = optim
		self.loader_train_s = loader_train_s
		self.loader_train_u = loader_train_u
		self.weak_augm_fn = weak_augm_fn
		self.strong_augm_fn = strong_augm_fn
		self.metrics_s = metrics_s
		self.metrics_u = metrics_u
		self.writer = writer
		self.criterion = criterion
		self.mode = mode
		self.threshold_multihot = threshold_multihot

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

			# Apply augmentations
			batch_s_weak = self.weak_augm_fn(batch_s)
			batch_u_weak = self.weak_augm_fn(batch_u)
			batch_u_strong = self.strong_augm_fn(batch_u)

			# Compute logits
			logits_s_weak = self.model(batch_s_weak)
			logits_u_weak = self.model(batch_u_weak)
			logits_u_strong = self.model(batch_u_strong)

			# Compute accuracies
			pred_s_weak = self.acti_fn(logits_s_weak, dim=1)
			pred_u_weak = self.acti_fn(logits_u_weak, dim=1)
			pred_u_strong = self.acti_fn(logits_u_strong, dim=1)

			if self.mode == "onehot":
				labels_u_guessed = binarize_onehot_labels(pred_u_weak)
			elif self.mode == "multihot":
				labels_u_guessed = (pred_u_weak > self.threshold_multihot).float()
			else:
				raise RuntimeError("Invalid argument \"mode = %s\". Use %s." % (self.mode, " or ".join(("onehot", "multihot"))))

			mean_acc_s = self.metrics_s(pred_s_weak, labels_s)
			mean_acc_u = self.metrics_u(pred_u_strong, labels_u_guessed)

			# Update model
			loss = self.criterion(pred_s_weak, labels_s, pred_u_weak, pred_u_strong, labels_u_guessed)
			self.optim.zero_grad()
			loss.backward()
			self.optim.step()

			# Store data
			losses.append(loss.item())
			acc_train_s.append(self.metrics_s.value.item())
			acc_train_u.append(self.metrics_u.value.item())

			print("Epoch {}, {:d}% \t loss: {:.4e} - acc_s: {:.4e} - acc_u: {:.4e} - took {:.4e}s".format(
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
