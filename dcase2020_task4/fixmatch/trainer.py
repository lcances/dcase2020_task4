import numpy as np

from easydict import EasyDict as edict
from time import time
from torch.nn import Module
from torch.nn.functional import one_hot
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable

from dcase2020.pytorch_metrics.metrics import Metrics

from dcase2020_task4.fixmatch.loss import FixMatchLoss
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
		weak_augm_fn: Callable,
		strong_augm_fn: Callable,
		metrics_s: Metrics,
		metrics_u: Metrics,
		writer: SummaryWriter,
		hparams: edict
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
		self.nb_classes = hparams.nb_classes
		self.mode = hparams.mode

		self.criterion = FixMatchLoss(
			lambda_u=hparams.lambda_u,
			threshold_mask=hparams.threshold_mask,
			threshold_multihot=hparams.threshold_multihot,
			mode=hparams.mode
		)

		self.pre_batch_fn = lambda batch: batch.cuda().float()
		if hparams.mode == "onehot":
			self.pre_label_fn = lambda label: one_hot(label.cuda().long(), self.nb_classes).float()
		elif hparams.mode == "multihot":
			self.pre_label_fn = lambda label: label.cuda().float()
		else:
			raise RuntimeError("Invalid argument \"mode = %s\". Use %s." % (hparams.mode, " or ".join(("onehot", "multihot"))))

	def train(self, epoch: int):
		train_start = time()
		self.metrics_s.reset()
		self.metrics_u.reset()
		self.model.train()

		losses, acc_train_s, acc_train_u = [], [], []
		loader_merged = MergeDataLoader([self.loader_train_s, self.loader_train_u])
		iter_train = iter(loader_merged)

		for i, (batch_s, labels_s, batch_u) in enumerate(iter_train):
			batch_s = self.pre_batch_fn(batch_s)
			batch_u = self.pre_batch_fn(batch_u)
			labels_s = self.pre_label_fn(labels_s)

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
				labels_u_guessed = (pred_u_weak > self.criterion.threshold_multihot).float()
			else:
				raise RuntimeError("Invalid argument \"mode = %s\". Use %s." % (self.mode, " or ".join(("onehot", "multihot"))))

			accuracy_s = self.metrics_s(pred_s_weak, labels_s)
			accuracy_u = self.metrics_u(pred_u_strong, labels_u_guessed)

			# Update model
			loss = self.criterion(pred_s_weak, labels_s, pred_u_weak, pred_u_strong, labels_u_guessed)
			self.optim.zero_grad()
			loss.backward()
			self.optim.step()

			# Store data
			losses.append(loss.item())
			acc_train_s.append(self.metrics_s.value.item())
			acc_train_u.append(self.metrics_u.value.item())

			print("Epoch {}, {:d}% \t loss: {:.4e} - acc_s: {:.4e} - acc_u: {:.4e} - took {:.2f}s".format(
				epoch + 1,
				int(100 * (i + 1) / len(loader_merged)),
				loss.item(),
				accuracy_s,
				accuracy_u,
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
