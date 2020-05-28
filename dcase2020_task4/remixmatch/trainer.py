import numpy as np
import torch

from easydict import EasyDict as edict
from time import time
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import one_hot
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable

from dcase2020.augmentation_utils.img_augmentations import Transform
from dcase2020.pytorch_metrics.metrics import Metrics

from dcase2020_task4.remixmatch.loss import ReMixMatchLoss
from dcase2020_task4.remixmatch.mixer import ReMixMatchMixer
from dcase2020_task4.trainer import SSTrainer
from dcase2020_task4.util.MergeDataLoader import MergeDataLoader
from dcase2020_task4.util.utils_match import get_lr


class ReMixMatchTrainer(SSTrainer):
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
		metrics_u1: Metrics,
		metrics_r: Metrics,
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
		self.metrics_u1 = metrics_u1
		self.metrics_r = metrics_r
		self.writer = writer
		self.nb_classes = hparams.nb_classes

		self.mixer = ReMixMatchMixer(
			model,
			weak_augm_fn,
			strong_augm_fn,
			hparams.nb_classes,
			hparams.nb_augms_strong,
			hparams.sharpen_temp,
			hparams.mixup_alpha,
			hparams.mode
		)
		self.criterion = ReMixMatchLoss(
			acti_fn,
			lambda_u=hparams.lambda_u,
			lambda_u1=hparams.lambda_u1,
			lambda_r=hparams.lambda_r,
			mode=hparams.mode,
		)

	def train(self, epoch: int):
		train_start = time()
		self.metrics_s.reset()
		self.metrics_u.reset()
		self.metrics_u1.reset()
		self.metrics_r.reset()
		self.model.train()

		angles_allowed = np.array([0.0, np.pi / 2.0, np.pi, -np.pi / 2.0])
		losses, acc_train_s, acc_train_u, acc_train_u1, acc_train_r = [], [], [], [], []
		loader_merged = MergeDataLoader([self.loader_train_s, self.loader_train_u])
		iter_train = iter(loader_merged)

		for i, (batch_s, labels_s, batch_u) in enumerate(iter_train):
			batch_s, batch_u = batch_s.cuda().float(), batch_u.cuda().float()
			labels_s = labels_s.cuda().long()
			labels_s = one_hot(labels_s, self.nb_classes).float()

			with torch.no_grad():
				self.mixer.distributions.add_batch_pred(labels_s, "labeled")
				pred_u = self.acti_fn(self.model(batch_u))
				self.mixer.distributions.add_batch_pred(pred_u, "unlabeled")

			# Apply mix
			batch_x_mixed, labels_x_mixed, batch_u_mixed, labels_u_mixed, batch_u1, labels_u1 = \
				self.mixer.mix(batch_s, labels_s, batch_u)

			# Predict labels for x (mixed), u (mixed) and u1 (strong augment)
			logits_x = self.model(batch_x_mixed)
			logits_u = self.model(batch_u_mixed)
			logits_u1 = self.model(batch_u1)

			# Rotate images and predict rotation for strong augment u1
			batch_u1_rotated, labels_r = apply_random_rot(batch_u1, angles_allowed)
			labels_r = one_hot(labels_r, len(angles_allowed)).float().cuda()
			logits_r = self.model.forward_rot(batch_u1_rotated)

			# Compute accuracies
			pred_x = self.acti_fn(logits_x)
			pred_u = self.acti_fn(logits_u)
			pred_u1 = self.acti_fn(logits_u1)
			pred_rot = self.acti_fn(logits_r)

			accuracy_s = self.metrics_s(pred_x, labels_x_mixed)
			accuracy_u = self.metrics_u(pred_u, labels_u_mixed)
			accuracy_u1 = self.metrics_u1(pred_u1, labels_u1)
			accuracy_rot = self.metrics_r(pred_rot, labels_r)

			# Update model
			loss = self.criterion(
				logits_x, labels_x_mixed,
				logits_u, labels_u_mixed,
				logits_u1, labels_u1,
				logits_r, labels_r,
			)

			self.optim.zero_grad()
			loss.backward()
			self.optim.step()

			# Store data
			losses.append(loss.item())
			acc_train_s.append(self.metrics_s.value.item())
			acc_train_u.append(self.metrics_u.value.item())
			acc_train_u1.append(self.metrics_u1.value.item())
			acc_train_r.append(self.metrics_r.value.item())

			print(
				"Epoch {}, {:d}% \t loss: {:.4e} - acc_s: {:.4e} - acc_u: {:.4e} - acc_u1: {:.4e} - acc_r: {:.4e} - took {:.2f}s".format(
					epoch + 1,
					int(100 * (i + 1) / len(loader_merged)),
					loss.item(),
					accuracy_s,
					accuracy_u,
					accuracy_u1,
					accuracy_rot,
					time() - train_start
				), end="\r")

		print("")

		self.writer.add_scalar("train/loss", float(np.mean(losses)), epoch)
		self.writer.add_scalar("train/acc_s", float(np.mean(acc_train_s)), epoch)
		self.writer.add_scalar("train/acc_u", float(np.mean(acc_train_u)), epoch)
		self.writer.add_scalar("train/acc_u1", float(np.mean(acc_train_u1)), epoch)
		self.writer.add_scalar("train/acc_rot", float(np.mean(acc_train_r)), epoch)
		self.writer.add_scalar("train/lr", get_lr(self.optim), epoch)

	def nb_examples_supervised(self) -> int:
		return len(self.loader_train_s) * self.loader_train_s.batch_size

	def nb_examples_unsupervised(self) -> int:
		return len(self.loader_train_u) * self.loader_train_u.batch_size


def apply_random_rot(batch: Tensor, angles_allowed) -> (Tensor, Tensor):
	idx = np.random.randint(0, len(angles_allowed), len(batch))
	angles = angles_allowed[idx]
	rotate_fn = lambda batch: torch.stack([
		Transform(1.0, rotation=(ang, ang))(x) for x, ang in zip(batch, angles)
	]).cuda()
	res = rotate_fn(batch)
	return res, torch.from_numpy(idx)
