import numpy as np
import torch

from time import time
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict

from metric_utils.metrics import Metrics

from dcase2020_task4.util.rampup import RampUp
from dcase2020_task4.trainer import SSTrainer
from dcase2020_task4.util.zip_cycle import ZipCycle
from dcase2020_task4.util.utils_match import get_lr


class MixMatchTrainer(SSTrainer):
	def __init__(
		self,
		model: Module,
		acti_fn: Callable,
		optim: Optimizer,
		loader_train_s_augm: DataLoader,
		loader_train_u_augms: DataLoader,
		metrics_s: Dict[str, Metrics],
		metrics_u: Dict[str, Metrics],
		criterion: Callable,
		writer: SummaryWriter,
		mixer: Callable,
		lambda_u_rampup: RampUp
	):
		self.model = model
		self.acti_fn = acti_fn
		self.optim = optim
		self.loader_train_s_augm = loader_train_s_augm
		self.loader_train_u_augms = loader_train_u_augms
		self.metrics_s = metrics_s
		self.metrics_u = metrics_u
		self.criterion = criterion
		self.writer = writer
		self.mixer = mixer
		self.lambda_u_rampup = lambda_u_rampup

	def train(self, epoch: int):
		train_start = time()
		self.model.train()
		self.reset_metrics()

		losses = []
		metric_values = {
			metric_name: [] for metric_name in (list(self.metrics_s.keys()) + list(self.metrics_u.keys()))
		}

		loaders_zip = ZipCycle([self.loader_train_s_augm, self.loader_train_u_augms])
		iter_train = iter(loaders_zip)

		for i, item in enumerate(iter_train):
			(s_batch_augm, s_labels_weak), u_batch_augms = item

			s_batch_augm = s_batch_augm.cuda().float()
			s_labels_weak = s_labels_weak.cuda().float()
			u_batch_augms = torch.stack(u_batch_augms).cuda().float()

			# Apply mix
			s_batch_mixed, s_labels_mixed, u_batch_mixed, u_labels_mixed = self.mixer(
				s_batch_augm, s_labels_weak, u_batch_augms
			)

			# Compute logits
			s_logits_mixed = self.model(s_batch_mixed)
			u_logits_mixed = self.model(u_batch_mixed)

			# Compute accuracies
			s_pred_mixed = self.acti_fn(s_logits_mixed, dim=1)
			u_pred_mixed = self.acti_fn(u_logits_mixed, dim=1)

			# Update model
			self.criterion.lambda_u = self.lambda_u_rampup.value()
			loss = self.criterion(s_pred_mixed, s_labels_mixed, u_pred_mixed, u_labels_mixed)
			self.optim.zero_grad()
			loss.backward()
			self.optim.step()

			self.lambda_u_rampup.step()

			# Compute metrics
			losses.append(loss.item())
			buffer = ["{:s}: {:.4e}".format("loss", np.mean(losses))]

			metric_pred_labels = [
				(self.metrics_s, s_pred_mixed, s_labels_mixed),
				(self.metrics_u, u_pred_mixed, u_labels_mixed),
			]
			for metrics, pred, labels in metric_pred_labels:
				for metric_name, metric in metrics.items():
					mean_s = metric(pred, labels)
					buffer.append("{:s}: {:.4e}".format(metric_name, mean_s))
					metric_values[metric_name].append(metric.value.item())

			buffer.append("took: {:.2f}s".format(time() - train_start))

			print("Epoch {:d}, {:d}% \t {:s}".format(
				epoch + 1,
				int(100 * (i + 1) / len(loaders_zip)),
				" - ".join(buffer),
			), end="\r")

		print("")

		self.writer.add_scalar("train/loss", float(np.mean(losses)), epoch)
		self.writer.add_scalar("train/lr", get_lr(self.optim), epoch)
		for metric_name, values in metric_values.items():
			self.writer.add_scalar("train/%s" % metric_name, float(np.mean(values)), epoch)

	def nb_examples_supervised(self) -> int:
		return len(self.loader_train_s_augm) * self.loader_train_s_augm.batch_size

	def nb_examples_unsupervised(self) -> int:
		return len(self.loader_train_u_augms) * self.loader_train_u_augms.batch_size

	def reset_metrics(self):
		metrics_lst = [self.metrics_s, self.metrics_u]
		for metrics in metrics_lst:
			for metric in metrics.values():
				metric.reset()
