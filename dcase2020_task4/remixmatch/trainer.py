import numpy as np
import torch

from time import time
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import one_hot
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict

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
		metrics_s: Dict[str, Metrics],
		metrics_u: Dict[str, Metrics],
		metrics_u1: Dict[str, Metrics],
		metrics_r: Dict[str, Metrics],
		criterion: Callable,
		writer: SummaryWriter,
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
		self.metrics_s = metrics_s
		self.metrics_u = metrics_u
		self.metrics_u1 = metrics_u1
		self.metrics_r = metrics_r
		self.criterion = criterion
		self.writer = writer
		self.mixer = mixer
		self.distributions = distributions
		self.rot_angles = rot_angles

		self.acti_fn_rot = lambda batch, dim: batch.softmax(dim=dim)

	def train(self, epoch: int):
		train_start = time()
		self.model.train()
		self.reset_metrics()

		losses = []
		metric_values = {metric_name: [] for metric_name in self.metrics_s.keys()}

		zip_cycle = ZipCycle([self.loader_train_s, self.loader_train_u])
		iter_train = iter(zip_cycle)

		for i, item in enumerate(iter_train):
			(s_batch_augm_strong, s_labels_weak), (u_batch_augm_weak, u_batch_augm_strongs) = item

			s_batch_augm_strong = s_batch_augm_strong.cuda().float()
			s_labels_weak = s_labels_weak.cuda().float()
			u_batch_augm_weak = u_batch_augm_weak.cuda().float()
			u_batch_augm_strongs = torch.stack(u_batch_augm_strongs).cuda().float()

			with torch.no_grad():
				self.distributions.add_batch_pred(s_labels_weak, "labeled")
				u_pred_augm_weak = self.acti_fn(self.model(u_batch_augm_weak), dim=1)
				self.distributions.add_batch_pred(u_pred_augm_weak, "unlabeled")

			# Apply mix
			s_batch_mixed, s_labels_mixed, u_batch_mixed, u_labels_mixed, u1_batch, u1_labels = \
				self.mixer(s_batch_augm_strong, s_labels_weak, u_batch_augm_weak, u_batch_augm_strongs)

			# Predict labels for x (mixed), u (mixed) and u1 (strong augment)
			s_logits_mixed = self.model(s_batch_mixed)
			u_logits_mixed = self.model(u_batch_mixed)
			u1_logits = self.model(u1_batch)

			s_pred_mixed = self.acti_fn(s_logits_mixed, dim=1)
			u_pred_mixed = self.acti_fn(u_logits_mixed, dim=1)
			u1_pred = self.acti_fn(u1_logits, dim=1)

			# Rotate images and predict rotation for strong augment u1
			u1_batch_rotated, r_labels = apply_random_rot(u1_batch, self.rot_angles)
			r_labels = one_hot(r_labels, len(self.rot_angles)).float().cuda()
			r_logits = self.model.forward_rot(u1_batch_rotated)
			r_pred = self.acti_fn_rot(r_logits, dim=1)

			# Update model
			loss = self.criterion(
				s_pred_mixed, s_labels_mixed,
				u_pred_mixed, u_labels_mixed,
				u1_pred, u1_labels,
				r_pred, r_labels,
			)
			self.optim.zero_grad()
			loss.backward()
			self.optim.step()

			# Compute accuracies
			losses.append(loss.item())
			buffer = ["{:s}: {:.4e}".format("loss", np.mean(losses))]

			metric_pred_labels = [
				(self.metrics_s, s_pred_mixed, s_labels_mixed),
				(self.metrics_u, u_pred_augm_weak, u_labels_mixed),
				(self.metrics_u1, u1_pred, u1_labels),
				(self.metrics_r, r_pred, r_labels),
			]
			for metrics, pred, labels in metric_pred_labels:
				for metric_name, metric in metrics.items():
					mean_s = metric(pred, labels)
					buffer.append("{:s}: {:.4e}".format(metric_name, mean_s))
					metric_values[metric_name].append(metric.value.item())

			buffer.append("took: {:.2f}s".format(time() - train_start))

			print("Epoch {:d}, {:d}% \t {:s}".format(
				epoch + 1,
				int(100 * (i + 1) / len(zip_cycle)),
				" - ".join(buffer),
			), end="\r")

		print("")

		self.writer.add_scalar("train/loss", np.mean(losses), epoch)
		self.writer.add_scalar("train/lr", get_lr(self.optim), epoch)
		for metric_name, values in metric_values.items():
			self.writer.add_scalar("train/%s" % metric_name, np.mean(values), epoch)

	def nb_examples_supervised(self) -> int:
		return len(self.loader_train_s) * self.loader_train_s.batch_size

	def nb_examples_unsupervised(self) -> int:
		return len(self.loader_train_u) * self.loader_train_u.batch_size

	def reset_metrics(self):
		metrics_lst = [self.metrics_s, self.metrics_u, self.metrics_u1, self.metrics_r]
		for metrics in metrics_lst:
			for metric in metrics.values():
				metric.reset()


def apply_random_rot(batch: Tensor, angles_allowed) -> (Tensor, Tensor):
	idx = np.random.randint(0, len(angles_allowed), len(batch))
	angles = angles_allowed[idx]
	rotate_fn = lambda batch: torch.stack([
		Transform(1.0, rotation=(ang, ang))(x) for x, ang in zip(batch, angles)
	]).cuda()
	res = rotate_fn(batch)
	return res, torch.from_numpy(idx)
