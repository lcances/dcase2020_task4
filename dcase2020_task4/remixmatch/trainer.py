import numpy as np
import torch

from time import time
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import one_hot
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable

from augmentation_utils.img_augmentations import Transform
from metric_utils.metrics import Metrics

from dcase2020_task4.remixmatch.model_distributions import ModelDistributions
from dcase2020_task4.trainer import SSTrainer
from dcase2020_task4.util.zip_cycle import ZipCycle
from dcase2020_task4.util.utils_match import get_lr


class ReMixMatchTrainer(SSTrainer):
	def __init__(
		self,
		model: Module,
		acti_fn: Callable,
		optim: Optimizer,
		loader_train_s: DataLoader,
		loader_train_u: DataLoader,
		metric_s: Metrics,
		metric_u: Metrics,
		metric_u1: Metrics,
		metric_r: Metrics,
		writer: SummaryWriter,
		criterion: Callable,
		mixer: Callable,
		distributions: ModelDistributions,
		rot_angles: np.array,
	):
		"""
			TODO : doc
			Note: model must implements torch.nn.Module and implements a method "forward_rot".
		"""
		self.model = model
		self.acti_fn = acti_fn
		self.optim = optim
		self.loader_train_s = loader_train_s
		self.loader_train_u = loader_train_u
		self.metric_s = metric_s
		self.metric_u = metric_u
		self.metric_u1 = metric_u1
		self.metric_r = metric_r
		self.writer = writer
		self.criterion = criterion
		self.mixer = mixer
		self.distributions = distributions
		self.rot_angles = rot_angles

		self.acti_fn_rot = lambda batch, dim: batch.softmax(dim=dim)

	def train(self, epoch: int):
		train_start = time()
		self.metric_s.reset()
		self.metric_u.reset()
		self.metric_u1.reset()
		self.metric_r.reset()
		self.model.train()

		losses, acc_train_s, acc_train_u, acc_train_u1, acc_train_r = [], [], [], [], []
		zip_cycle = ZipCycle([self.loader_train_s, self.loader_train_u])
		iter_train = iter(zip_cycle)

		for i, ((batch_s_strong, labels_s), (batch_u_weak, batch_u_strongs)) in enumerate(iter_train):
			batch_s_strong = batch_s_strong.cuda().float()
			labels_s = labels_s.cuda().float()
			batch_u_weak = batch_u_weak.cuda().float()
			batch_u_strongs = torch.stack(batch_u_strongs).cuda().float()

			with torch.no_grad():
				self.distributions.add_batch_pred(labels_s, "labeled")
				pred_u = self.acti_fn(self.model(batch_u_weak), dim=1)
				self.distributions.add_batch_pred(pred_u, "unlabeled")

			# Apply mix
			batch_s_mixed, labels_s_mixed, batch_u_mixed, labels_u_mixed, batch_u1, labels_u1 = \
				self.mixer(batch_s_strong, labels_s, batch_u_weak, batch_u_strongs)

			# Predict labels for x (mixed), u (mixed) and u1 (strong augment)
			logits_s = self.model(batch_s_mixed)
			logits_u = self.model(batch_u_mixed)
			logits_u1 = self.model(batch_u1)

			# Rotate images and predict rotation for strong augment u1
			batch_u1_rotated, labels_r = apply_random_rot(batch_u1, self.rot_angles)
			labels_r = one_hot(labels_r, len(self.rot_angles)).float().cuda()
			logits_r = self.model.forward_rot(batch_u1_rotated)

			pred_s = self.acti_fn(logits_s, dim=1)
			pred_u = self.acti_fn(logits_u, dim=1)
			pred_u1 = self.acti_fn(logits_u1, dim=1)
			pred_r = self.acti_fn_rot(logits_r, dim=1)

			# Compute accuracies
			mean_acc_s = self.metric_s(pred_s, labels_s_mixed)
			mean_acc_u = self.metric_u(pred_u, labels_u_mixed)
			mean_acc_u1 = self.metric_u1(pred_u1, labels_u1)
			mean_acc_r = self.metric_r(pred_r, labels_r)

			# Update model
			loss = self.criterion(
				pred_s, labels_s_mixed,
				pred_u, labels_u_mixed,
				pred_u1, labels_u1,
				pred_r, labels_r,
			)

			self.optim.zero_grad()
			loss.backward()
			self.optim.step()

			# Store data
			losses.append(loss.item())
			acc_train_s.append(self.metric_s.value.item())
			acc_train_u.append(self.metric_u.value.item())
			acc_train_u1.append(self.metric_u1.value.item())
			acc_train_r.append(self.metric_r.value.item())

			print(
				"Epoch {}, {:d}% \t loss: {:.4e} - acc_s: {:.4e} - acc_u: {:.4e} - acc_u1: {:.4e} - acc_r: {:.4e} - took {:.2f}s".format(
					epoch + 1,
					int(100 * (i + 1) / len(zip_cycle)),
					loss.item(),
					mean_acc_s,
					mean_acc_u,
					mean_acc_u1,
					mean_acc_r,
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
