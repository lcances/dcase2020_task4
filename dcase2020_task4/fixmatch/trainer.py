import numpy as np

from time import time
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict

from metric_utils.metrics import Metrics

from dcase2020_task4.util.zip_cycle import ZipCycle
from dcase2020_task4.util.utils_match import binarize_onehot_labels, get_lr
from dcase2020_task4.trainer import SSTrainer


class FixMatchTrainer(SSTrainer):
	def __init__(
		self,
		model: Module,
		acti_fn: Callable,
		optim: Optimizer,
		loader_train_s_weak: DataLoader,
		loader_train_u_weak_strong: DataLoader,
		metrics_s: Dict[str, Metrics],
		metrics_u: Dict[str, Metrics],
		writer: SummaryWriter,
		criterion: Callable,
		mode: str = "onehot",
		threshold_multihot: float = 0.5,
	):
		self.model = model
		self.acti_fn = acti_fn
		self.optim = optim
		self.loader_train_s_weak = loader_train_s_weak
		self.loader_train_u_weak_strong = loader_train_u_weak_strong
		self.metrics_s = metrics_s
		self.metrics_u = metrics_u
		self.writer = writer
		self.criterion = criterion
		self.mode = mode
		self.threshold_multihot = threshold_multihot

	def train(self, epoch: int):
		train_start = time()
		self.model.train()
		self.reset_metrics()

		losses = []
		metric_values = {
			metric_name: [] for metric_name in (list(self.metrics_s.keys()) + list(self.metrics_u.keys()))
		}

		zip_cycle = ZipCycle([self.loader_train_s_weak, self.loader_train_u_weak_strong])
		iter_train = iter(zip_cycle)

		for i, item in enumerate(iter_train):
			(batch_s_weak, labels_s), (batch_u_weak, batch_u_strong) = item

			batch_s_weak = batch_s_weak.cuda().float()
			labels_s = labels_s.cuda().float()
			batch_u_weak = batch_u_weak.cuda().float()
			batch_u_strong = batch_u_strong.cuda().float()

			# Compute logits
			logits_s_weak = self.model(batch_s_weak)
			logits_u_weak = self.model(batch_u_weak)
			logits_u_strong = self.model(batch_u_strong)

			# Compute accuracies
			pred_s_weak = self.acti_fn(logits_s_weak, dim=1)
			pred_u_weak = self.acti_fn(logits_u_weak, dim=1)
			pred_u_strong = self.acti_fn(logits_u_strong, dim=1)

			if self.mode == "onehot":
				labels_u_guessed = binarize_onehot_labels(pred_u_weak)
			elif self.mode == "multihot":
				labels_u_guessed = (pred_u_weak > self.threshold_multihot).float()
			else:
				raise RuntimeError("Invalid argument \"mode = %s\". Use %s." % (self.mode, " or ".join(("onehot", "multihot"))))

			# Update model
			loss = self.criterion(pred_s_weak, labels_s, pred_u_weak, pred_u_strong, labels_u_guessed)
			self.optim.zero_grad()
			loss.backward()
			self.optim.step()

			losses.append(loss.item())
			buffer = ["{:s}: {:.4e}".format("loss", np.mean(losses))]

			metric_pred_labels = [
				(self.metrics_s, pred_s_weak, labels_s),
				(self.metrics_u, pred_u_strong, labels_u_guessed),
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
				" - ".join(buffer)
			), end="\r")

		print("")

		self.writer.add_scalar("train/loss", float(np.mean(losses)), epoch)
		self.writer.add_scalar("train/lr", get_lr(self.optim), epoch)
		for metric_name, values in metric_values.items():
			self.writer.add_scalar("train/%s" % metric_name, float(np.mean(values)), epoch)

	def nb_examples_supervised(self) -> int:
		return len(self.loader_train_s_weak) * self.loader_train_s_weak.batch_size

	def nb_examples_unsupervised(self) -> int:
		return len(self.loader_train_u_weak_strong) * self.loader_train_u_weak_strong.batch_size

	def reset_metrics(self):
		metrics_lst = [self.metrics_s, self.metrics_u]
		for metrics in metrics_lst:
			for metric in metrics.values():
				metric.reset()
