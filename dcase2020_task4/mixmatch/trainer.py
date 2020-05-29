import numpy as np

from easydict import EasyDict as edict
from time import time
from torch.nn import Module
from torch.nn.functional import one_hot
from torch.nn.utils import clip_grad_norm_
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable

from dcase2020.pytorch_metrics.metrics import Metrics

from dcase2020_task4.mixmatch.loss import MixMatchLoss
from dcase2020_task4.mixmatch.mixer import MixMatchMixer
from dcase2020_task4.mixmatch.RampUp import RampUp
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
		hparams: edict
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
		self.nb_classes = hparams.nb_classes

		nb_rampup_steps = hparams.nb_epochs * len(loader_train_u)

		self.lambda_u_rampup = RampUp(max_value=hparams.lambda_u_max, nb_steps=nb_rampup_steps)
		self.mixer = MixMatchMixer(model, augm_fn, hparams.nb_augms, hparams.sharpen_temp, hparams.mixup_alpha)
		self.criterion = MixMatchLoss(
			acti_fn, lambda_u=hparams.lambda_u_max, mode=hparams.mode, criterion_unsupervised=hparams.criterion_unsupervised
		)

	def train(self, epoch: int):
		train_start = time()
		self.metrics_s.reset()
		self.metrics_u.reset()
		self.model.train()

		losses, acc_train_s, acc_train_u = [], [], []
		loader_merged = MergeDataLoader([self.loader_train_s, self.loader_train_u])
		iter_train = iter(loader_merged)

		for i, (batch_s, labels_s, batch_u) in enumerate(iter_train):
			batch_s, batch_u = batch_s.cuda().float(), batch_u.cuda().float()
			labels_s = labels_s.cuda().long()
			labels_s = one_hot(labels_s, self.nb_classes).float()

			# Apply mix
			batch_s_mixed, labels_s_mixed, batch_u_mixed, labels_u_mixed = self.mixer.mix(batch_s, labels_s, batch_u)

			# Compute logits
			logits_x = self.model(batch_s_mixed)
			logits_u = self.model(batch_u_mixed)

			# Compute accuracies
			pred_x = self.acti_fn(logits_x)
			pred_u = self.acti_fn(logits_u)

			accuracy_s = self.metrics_s(pred_x, labels_s_mixed)
			accuracy_u = self.metrics_u(pred_u, labels_u_mixed)

			# Update model
			self.criterion.lambda_u = self.lambda_u_rampup.value()
			loss = self.criterion(pred_x, labels_s_mixed, pred_u, labels_u_mixed)
			self.optim.zero_grad()
			loss.backward()
			# clip_grad_norm_(self.model.parameters(), 100)  # TODO : rem
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
