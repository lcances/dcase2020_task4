import numpy as np
import torch

from torch import Tensor
from torch.nn import Module
from torch.nn.functional import one_hot
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict, List, Optional

from augmentation_utils.img_augmentations import Transform
from metric_utils.metrics import Metrics

from dcase2020_task4.metrics_recorder import MetricsRecorder
from dcase2020_task4.remixmatch.losses.abc import ReMixMatchLossTagABC
from dcase2020_task4.trainer_abc import SSTrainerABC

from dcase2020_task4.util.avg_distributions import AvgDistributions
from dcase2020_task4.util.sharpen import sharpen, sharpen_multi
from dcase2020_task4.util.utils_match import get_lr
from dcase2020_task4.util.zip_cycle import ZipCycle


class ReMixMatchTrainer(SSTrainerABC):
	def __init__(
		self,
		model: Module,
		acti_fn: Callable,
		acti_rot_fn: Callable,
		optim: Optimizer,
		loader_train_s: DataLoader,
		loader_train_u: DataLoader,
		metrics_s: Dict[str, Metrics],
		metrics_u: Dict[str, Metrics],
		metrics_u1: Dict[str, Metrics],
		metrics_r: Dict[str, Metrics],
		criterion: ReMixMatchLossTagABC,
		writer: Optional[SummaryWriter],
		mixer: Callable,
		distributions: Optional[AvgDistributions],
		rot_angles: np.array,
		sharpen_fn: Callable,
	):
		"""
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
		self.sharpen_fn = sharpen_fn

		self.acti_rot_fn = acti_rot_fn
		self.metrics_recorder = MetricsRecorder(
			"train/",
			list(self.metrics_s.keys()) +
			list(self.metrics_u.keys()) +
			list(self.metrics_u1.keys()) +
			list(self.metrics_r.keys()) +
			["loss", "loss_s", "loss_u", "loss_u1", "loss_r"]
		)

	def train(self, epoch: int):
		self.reset_all_metrics()
		self.metrics_recorder.reset_epoch()
		self.model.train()

		loaders_zip = ZipCycle([self.loader_train_s, self.loader_train_u])
		iter_train = iter(loaders_zip)

		for i, item in enumerate(iter_train):
			(s_batch_augm_strong, s_labels_weak), (u_batch_augm_weak, u_batch_augm_strongs) = item

			s_batch_augm_strong = s_batch_augm_strong.cuda().float()
			s_labels_weak = s_labels_weak.cuda().float()
			u_batch_augm_weak = u_batch_augm_weak.cuda().float()
			u_batch_augm_strongs = torch.stack(u_batch_augm_strongs).cuda().float()

			with torch.no_grad():
				# Compute guessed label
				u_logits_weak = self.model(u_batch_augm_weak)
				u_pred_augm_weak = self.acti_fn(u_logits_weak, dim=1)

				if self.distributions is not None:
					self.distributions.add_batch_pred(s_labels_weak, "labeled")
					self.distributions.add_batch_pred(u_pred_augm_weak, "unlabeled")
					u_label_guessed = self.distributions.apply_distribution_alignment(u_pred_augm_weak, dim=1)
				else:
					u_label_guessed = u_pred_augm_weak

				u_label_guessed = self.sharpen_fn(u_label_guessed, dim=1)

				# Apply mix
				s_batch_mixed, s_labels_mixed, u_batch_mixed, u_labels_mixed, u1_batch, u1_labels = \
					self.mixer(s_batch_augm_strong, s_labels_weak, u_batch_augm_weak, u_batch_augm_strongs, u_label_guessed)

			# Rotate images
			u1_batch_rotated, r_labels = apply_random_rot(u1_batch, self.rot_angles)
			r_labels = one_hot(r_labels, len(self.rot_angles)).float().cuda()

			# Predict labels for x (mixed), u (mixed) and u1 (strong augment)
			s_logits_mixed = self.model(s_batch_mixed)
			u_logits_mixed = self.model(u_batch_mixed)
			u1_logits = self.model(u1_batch)

			s_pred_mixed = self.acti_fn(s_logits_mixed, dim=1)
			u_pred_mixed = self.acti_fn(u_logits_mixed, dim=1)
			u1_pred = self.acti_fn(u1_logits, dim=1)

			# Predict rotation for strong augment u1
			r_logits = self.model.forward_rot(u1_batch_rotated)
			r_pred = self.acti_rot_fn(r_logits, dim=1)

			# Update model
			loss, loss_s, loss_u, loss_u1, loss_r = self.criterion(
				s_pred_mixed, s_labels_mixed,
				u_pred_mixed, u_labels_mixed,
				u1_pred, u1_labels,
				r_pred, r_labels
			)
			self.optim.zero_grad()
			loss.backward()
			self.optim.step()

			# Compute metrics
			with torch.no_grad():
				self.metrics_recorder.add_value("loss", loss.item())
				self.metrics_recorder.add_value("loss_s", loss_s.item())
				self.metrics_recorder.add_value("loss_u", loss_u.item())
				self.metrics_recorder.add_value("loss_u1", loss_u1.item())
				self.metrics_recorder.add_value("loss_r", loss_r.item())

				metrics_preds_labels = [
					(self.metrics_s, s_pred_mixed, s_labels_mixed),
					(self.metrics_u, u_pred_mixed, u_labels_mixed),
					(self.metrics_u1, u1_pred, u1_labels),
					(self.metrics_r, r_pred, r_labels),
				]

				self.metrics_recorder.apply_metrics_and_add(metrics_preds_labels)
				self.metrics_recorder.print_metrics(epoch, i, len(loaders_zip))

		print("")

		if self.writer is not None:
			self.writer.add_scalar("hparams/lr", get_lr(self.optim), epoch)
			self.writer.add_scalar("hparams/lambda_u", self.criterion.lambda_u, epoch)
			self.writer.add_scalar("hparams/lambda_u1", self.criterion.lambda_u1, epoch)
			self.writer.add_scalar("hparams/lambda_r", self.criterion.lambda_r, epoch)
			self.metrics_recorder.store_in_writer(self.writer, epoch)

	def nb_examples_supervised(self) -> int:
		return len(self.loader_train_s) * self.loader_train_s.batch_size

	def nb_examples_unsupervised(self) -> int:
		return len(self.loader_train_u) * self.loader_train_u.batch_size

	def get_all_metrics(self) -> List[Dict[str, Metrics]]:
		return [self.metrics_s, self.metrics_u, self.metrics_u1, self.metrics_r]


def apply_random_rot(batch: Tensor, angles_allowed) -> (Tensor, Tensor):
	indexes = np.random.randint(0, len(angles_allowed), len(batch))
	angles = angles_allowed[indexes]
	res = torch.stack([
		Transform(1.0, rotation=(ang, ang))(x) for x, ang in zip(batch, angles)
	]).cuda()
	return res, torch.from_numpy(indexes)
