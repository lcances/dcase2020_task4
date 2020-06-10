import numpy as np

from time import time
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict

from metric_utils.metrics import Metrics
from dcase2020_task4.util.utils_match import get_lr
from dcase2020_task4.trainer import Trainer


class SupervisedTrainer(Trainer):
	def __init__(
		self,
		model: Module,
		acti_fn: Callable,
		optim: Optimizer,
		loader: DataLoader,
		criterion: Callable,
		metrics: Dict[str, Metrics],
		writer: SummaryWriter,
	):
		self.model = model
		self.acti_fn = acti_fn
		self.optim = optim
		self.loader = loader
		self.criterion = criterion
		self.metrics = metrics
		self.writer = writer

	def train(self, epoch: int):
		train_start = time()
		self.model.train()
		self.reset_metrics()

		losses = []
		metric_values = {metric_name: [] for metric_name in self.metrics.keys()}

		iter_train = iter(self.loader)

		for i, (x, y) in enumerate(iter_train):
			x = x.cuda().float()
			y = y.cuda().float()

			# Compute logits
			logits = self.model(x)
			pred = self.acti_fn(logits, dim=1)

			# Update model
			loss = self.criterion(pred, y).mean()
			self.optim.zero_grad()
			loss.backward()
			self.optim.step()

			# Compute accuracies
			losses.append(loss.item())
			buffer = ["{:s}: {:.4e}".format("loss", np.mean(losses))]

			metric_pred_labels = [
				(self.metrics, pred, y),
			]
			for metrics, pred, labels in metric_pred_labels:
				for metric_name, metric in metrics.items():
					mean_s = metric(pred, labels)
					buffer.append("%s: %.4e" % (metric_name, mean_s))
					metric_values[metric_name].append(metric.value.item())

			buffer.append("took: %.2fs" % (time() - train_start))

			print("Epoch {:d}, {:d}% \t {:s}".format(
				epoch + 1,
				int(100 * (i + 1) / len(self.loader)),
				" - ".join(buffer),
			), end="\r")

		print("")

		self.writer.add_scalar("train/loss", float(np.mean(losses)), epoch)
		self.writer.add_scalar("train/lr", get_lr(self.optim), epoch)
		for metric_name, values in metric_values.items():
			self.writer.add_scalar("train/%s" % metric_name, float(np.mean(values)), epoch)

	def nb_examples(self) -> int:
		return len(self.loader) * self.loader.batch_size

	def reset_metrics(self):
		metrics_lst = [self.metrics]
		for metrics in metrics_lst:
			for metric in metrics.values():
				metric.reset()
