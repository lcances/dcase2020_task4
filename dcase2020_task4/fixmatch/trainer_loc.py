import numpy as np
import torch

from time import time
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict, Optional

from metric_utils.metrics import Metrics

from dcase2020_task4.fixmatch.losses.abc import FixMatchLossMultiHotLocABC
from dcase2020_task4.util.utils_match import get_lr
from dcase2020_task4.util.zip_cycle import ZipCycle
from dcase2020_task4.trainer import SSTrainer


class FixMatchTrainerLoc(SSTrainer):
	def __init__(
		self,
		model: Module,
		acti_fn: Callable,
		optim: Optimizer,
		loader_train_s_augm_weak: DataLoader,
		loader_train_u_augms_weak_strong: DataLoader,
		metrics_s_weak: Dict[str, Metrics],
		metrics_u_weak: Dict[str, Metrics],
		metrics_s_strong: Dict[str, Metrics],
		metrics_u_strong: Dict[str, Metrics],
		criterion: FixMatchLossMultiHotLocABC,
		writer: Optional[SummaryWriter],
		threshold_multihot: float,
	):
		self.model = model
		self.acti_fn = acti_fn
		self.optim = optim
		self.loader_train_s_augm_weak = loader_train_s_augm_weak
		self.loader_train_u_augms_weak_strong = loader_train_u_augms_weak_strong
		self.metrics_s_weak = metrics_s_weak
		self.metrics_u_weak = metrics_u_weak
		self.metrics_s_strong = metrics_s_strong
		self.metrics_u_strong = metrics_u_strong
		self.criterion = criterion
		self.writer = writer
		self.threshold_multihot = threshold_multihot

	def train(self, epoch: int):
		train_start = time()
		self.model.train()
		self.reset_metrics()

		metric_values = {
			metric_name: [] for metric_name in (
				list(self.metrics_s_weak.keys()) +
				list(self.metrics_u_weak.keys()) +
				list(self.metrics_s_strong.keys()) +
				list(self.metrics_u_strong.keys()) +
				["loss", "loss_s_weak", "loss_u_weak", "loss_s_strong", "loss_u_strong"]
			)
		}

		loaders_zip = ZipCycle([self.loader_train_s_augm_weak, self.loader_train_u_augms_weak_strong])
		iter_train = iter(loaders_zip)

		for i, item in enumerate(iter_train):
			(s_batch_augm_weak, s_labels_weak, s_labels_strong), (u_batch_augm_weak, u_batch_augm_strong) = item

			s_batch_augm_weak = s_batch_augm_weak.cuda().float()
			s_labels_weak = s_labels_weak.cuda().float()
			s_labels_strong = s_labels_strong.cuda().float()
			u_batch_augm_weak = u_batch_augm_weak.cuda().float()
			u_batch_augm_strong = u_batch_augm_strong.cuda().float()

			# Compute logits
			s_logits_weak_augm_weak, s_logits_strong_augm_weak = self.model(s_batch_augm_weak)
			u_logits_weak_augm_strong, u_logits_strong_augm_strong = self.model(u_batch_augm_strong)

			s_pred_weak_augm_weak = self.acti_fn(s_logits_weak_augm_weak, dim=1)
			u_pred_weak_augm_strong = self.acti_fn(u_logits_weak_augm_strong, dim=1)

			s_pred_strong_augm_weak = self.acti_fn(s_logits_strong_augm_weak, dim=(1, 2))
			u_pred_strong_augm_strong = self.acti_fn(u_logits_strong_augm_strong, dim=(1, 2))

			with torch.no_grad():
				u_logits_weak_augm_weak, u_logits_strong_augm_weak = self.model(u_batch_augm_weak)
				u_pred_weak_augm_weak = self.acti_fn(u_logits_weak_augm_weak, dim=1)
				u_pred_strong_augm_weak = self.acti_fn(u_logits_strong_augm_weak, dim=(1, 2))

				# Use guess u label with prediction of weak augmentation of u
				u_labels_weak_guessed = (u_pred_weak_augm_weak > self.threshold_multihot).float()
				u_labels_strong_guessed = (u_pred_strong_augm_weak > self.threshold_multihot).float()

			# Update model
			loss, loss_s_weak, loss_u_weak, loss_s_strong, loss_u_strong = self.criterion(
				s_pred_weak_augm_weak, s_labels_weak,
				u_pred_weak_augm_weak, u_pred_weak_augm_strong, u_labels_weak_guessed,
				s_pred_strong_augm_weak, s_labels_strong,
				u_pred_strong_augm_weak, u_pred_strong_augm_strong, u_labels_strong_guessed,
			)
			self.optim.zero_grad()
			loss.backward()
			self.optim.step()

			with torch.no_grad():
				metric_values["loss"].append(loss.item())
				metric_values["loss_s_weak"].append(loss_s_weak.item())
				metric_values["loss_u_weak"].append(loss_u_weak.item())
				metric_values["loss_s_strong"].append(loss_s_strong.item())
				metric_values["loss_u_strong"].append(loss_u_strong.item())

				metric_pred_labels = [
					(self.metrics_s_weak, s_pred_weak_augm_weak, s_labels_weak),
					(self.metrics_u_weak, u_pred_weak_augm_strong, u_labels_weak_guessed),
					(self.metrics_s_strong, s_pred_strong_augm_weak, s_labels_strong),
					(self.metrics_u_strong, u_pred_strong_augm_strong, u_labels_strong_guessed),
				]
				for metrics, pred, labels in metric_pred_labels:
					for metric_name, metric in metrics.items():
						_mean_s = metric(pred, labels)
						metric_values[metric_name].append(metric.value.item())

				prints_buffer = [
					"{:s}: {:.4e}".format(name, np.mean(values))
					for name, values in metric_values.items()
				]
				prints_buffer.append("took: {:.2f}s".format(time() - train_start))

				print("Epoch {:d}, {:d}% \t {:s}".format(
					epoch + 1,
					int(100 * (i + 1) / len(loaders_zip)),
					" - ".join(prints_buffer)
				), end="\r")

		print("")

		if self.writer is not None:
			self.writer.add_scalar("train/lr", get_lr(self.optim), epoch)
			for metric_name, values in metric_values.items():
				self.writer.add_scalar("train/%s" % metric_name, float(np.mean(values)), epoch)

	def nb_examples_supervised(self) -> int:
		return len(self.loader_train_s_augm_weak) * self.loader_train_s_augm_weak.batch_size

	def nb_examples_unsupervised(self) -> int:
		return len(self.loader_train_u_augms_weak_strong) * self.loader_train_u_augms_weak_strong.batch_size

	def reset_metrics(self):
		metrics_lst = [self.metrics_s_weak, self.metrics_u_weak, self.metrics_s_strong, self.metrics_u_strong]
		for metrics in metrics_lst:
			for metric in metrics.values():
				metric.reset()
