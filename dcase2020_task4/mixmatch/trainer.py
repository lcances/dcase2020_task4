import numpy as np
import torch

from time import time
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict

from metric_utils.metrics import Metrics

from dcase2020_task4.mixmatch.rampup import RampUp
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
		writer: SummaryWriter,
		criterion: Callable,
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
		self.writer = writer
		self.criterion = criterion
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

		zip_cycle = ZipCycle([self.loader_train_s_augm, self.loader_train_u_augms])
		iter_train = iter(zip_cycle)

		for i, item in enumerate(iter_train):
			(batch_s_augm, labels_s), batch_u_augms = item

			batch_s_augm = batch_s_augm.cuda().float()
			labels_s = labels_s.cuda().float()
			batch_u_augms = torch.stack(batch_u_augms).cuda().float()

			# Apply mix
			batch_s_mixed, labels_s_mixed, batch_u_mixed, labels_u_mixed = self.mixer(batch_s_augm, labels_s, batch_u_augms)

			# Compute logits
			logits_s_mixed = self.model(batch_s_mixed)
			logits_u_mixed = self.model(batch_u_mixed)

			# Compute accuracies
			pred_s_mixed = self.acti_fn(logits_s_mixed, dim=1)
			pred_u_mixed = self.acti_fn(logits_u_mixed, dim=1)

			# Update model
			self.criterion.lambda_u = self.lambda_u_rampup.value()
			loss = self.criterion(pred_s_mixed, labels_s_mixed, pred_u_mixed, labels_u_mixed)
			self.optim.zero_grad()
			loss.backward()
			self.optim.step()

			self.lambda_u_rampup.step()

			# Compute metrics
			losses.append(loss.item())
			buffer = ["{:s}: {:.4e}".format("loss", np.mean(losses))]

			metric_pred_labels = [
				(self.metrics_s, pred_s_mixed, labels_s_mixed),
				(self.metrics_u, pred_u_mixed, labels_u_mixed),
			]
			for metrics, pred, labels in metric_pred_labels:
				for metric_name, metric in metrics.items():
					mean_s = metric(pred, labels)
					buffer.append("%s: %.4e" % (metric_name, mean_s))
					metric_values[metric_name].append(metric.value.item())

			buffer.append("took: %.2fs" % (time() - train_start))

			print("Epoch {:d}, {:d}% \t {:s}".format(
				epoch + 1,
				int(100 * (i + 1) / len(zip_cycle)),
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
