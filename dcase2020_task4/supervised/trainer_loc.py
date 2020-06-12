import numpy as np
import torch

from time import time
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict

from metric_utils.metrics import Metrics
from dcase2020_task4.util.utils_match import get_lr
from dcase2020_task4.trainer import Trainer


class SupervisedTrainerLoc(Trainer):
	def __init__(
		self,
		model: Module,
		acti_fn: Callable,
		optim: Optimizer,
		loader: DataLoader,
		metrics_weak: Dict[str, Metrics],
		metrics_strong: Dict[str, Metrics],
		criterion: Callable,
		writer: SummaryWriter
	):
		self.model = model
		self.acti_fn = acti_fn
		self.optim = optim
		self.loader = loader
		self.metrics_weak = metrics_weak
		self.metrics_strong = metrics_strong
		self.criterion = criterion
		self.writer = writer

	def train(self, epoch: int):
		train_start = time()
		self.model.train()
		self.reset_metrics()

		metric_values = {
			metric_name: [] for metric_name in (
				list(self.metrics_weak.keys()) +
				list(self.metrics_strong.keys()) +
				["loss", "loss_weak", "loss_strong"]
			)
		}
		iter_val = iter(self.loader)

		for i, (batch, labels_weak, labels_strong) in enumerate(iter_val):
			batch = batch.cuda().float()
			labels_weak = labels_weak.cuda().float()
			labels_strong = labels_strong.cuda().float()

			logits_weak, logits_strong = self.model(batch)

			pred_weak = self.acti_fn(logits_weak, dim=1)
			pred_strong = self.acti_fn(logits_strong, dim=1)

			loss_weak, loss_strong, loss = self.criterion(logits_weak, logits_strong, labels_weak, labels_strong)
			self.optim.zero_grad()
			loss.backward()
			self.optim.step()

			metric_values["loss"].append(loss.item())
			metric_values["loss_weak"].append(loss_weak.item())
			metric_values["loss_strong"].append(loss_strong.item())

			metric_pred_labels = [
				(self.metrics_weak, pred_weak, labels_weak),
				(self.metrics_strong, pred_strong, labels_strong),
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
				int(100 * (i + 1) / len(self.loader)),
				" - ".join(prints_buffer)
			), end="\r")

		print("")

		if self.writer is not None:
			self.writer.add_scalar("train/lr", get_lr(self.optim), epoch)
			for metric_name, values in metric_values.items():
				self.writer.add_scalar("train/%s" % metric_name, np.mean(values), epoch)

	def nb_examples(self) -> int:
		return len(self.loader) * self.loader.batch_size

	def reset_metrics(self):
		metrics_lst = [self.metrics_weak, self.metrics_strong]
		for metrics in metrics_lst:
			for metric in metrics.values():
				metric.reset()
