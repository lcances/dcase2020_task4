import numpy as np
import torch

from abc import ABC
from time import time
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict

from metric_utils.metrics import Metrics


class ValidatorABC(ABC):
	def val(self, epoch: int):
		raise NotImplementedError("Abstract method")

	def nb_examples(self) -> int:
		raise NotImplementedError("Abstract method")


class DefaultValidator(ValidatorABC):
	def __init__(
		self,
		model: Module,
		acti_fn: Callable,
		loader: DataLoader,
		metrics: Dict[str, Metrics],
		writer: SummaryWriter
	):
		self.model = model
		self.acti_fn = acti_fn
		self.loader = loader
		self.metrics = metrics
		self.writer = writer

	def val(self, epoch: int):
		with torch.no_grad():
			val_start = time()
			self.model.eval()
			self.reset_metrics()

			metrics_values_dict = {metric_name: [] for metric_name in self.metrics.keys()}
			iter_val = iter(self.loader)

			for i, (x, y) in enumerate(iter_val):
				x = x.cuda().float()
				y = y.cuda().float()

				logits_x = self.model(x)
				pred_x = self.acti_fn(logits_x, dim=1)

				buffer = []

				# Compute metrics and store them in buffer for print and in metrics_values_dict for writer
				for metric_name, metric in self.metrics.items():
					cur_mean_value = metric(pred_x, y)
					buffer.append("%s: %.4e" % (metric_name, cur_mean_value))
					metrics_values_dict[metric_name].append(metric.value.item())

				# Add time elapsed since the beginning of validation
				buffer.append("took %.2fs" % (time() - val_start))

				# Print buffer
				print("Epoch {}, {:d}% \t {:s}".format(
					epoch + 1,
					int(100 * (i + 1) / len(self.loader)),
					" - ".join(buffer)
				), end="\r")

			print("")

			for metric_name, metric_values in metrics_values_dict.items():
				self.writer.add_scalar("val/%s" % metric_name, float(np.mean(metric_values)), epoch)

	def nb_examples(self) -> int:
		return len(self.loader) * self.loader.batch_size

	def reset_metrics(self):
		for metric in self.metrics.values():
			metric.reset()


class DefaultValidatorLoc(ValidatorABC):
	def __init__(
		self,
		model: Module,
		acti_fn: Callable,
		loader: DataLoader,
		metrics_weak: Dict[str, Metrics],
		metrics_strong: Dict[str, Metrics],
		writer: SummaryWriter
	):
		self.model = model
		self.acti_fn = acti_fn
		self.loader = loader
		self.metrics_weak = metrics_weak
		self.metrics_strong = metrics_strong
		self.writer = writer

	def val(self, epoch: int):
		with torch.no_grad():
			val_start = time()
			self.model.eval()
			self.reset_metrics()

			metric_values = {
				metric_name: [] for metric_name in (
					list(self.metrics_weak.keys()) + list(self.metrics_strong.keys())
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

				mask = torch.clamp(labels_strong.sum(dim=(1, 2)), 0, 1)
				mask_nums = []
				for j, has_strong_label in enumerate(mask):
					if has_strong_label != 0:
						mask_nums.append(j)
				pred_strong = pred_strong[mask_nums]
				labels_strong = labels_strong[mask_nums]

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
				prints_buffer.append("took: {:.2f}s".format(time() - val_start))

				print("Epoch {:d}, {:d}% \t {:s}".format(
					epoch + 1,
					int(100 * (i + 1) / len(self.loader)),
					" - ".join(prints_buffer)
				), end="\r")

			print("")

			for metric_name, values in metric_values.items():
				self.writer.add_scalar("val/%s" % metric_name, np.mean(values), epoch)

	def nb_examples(self) -> int:
		return len(self.loader) * self.loader.batch_size

	def reset_metrics(self):
		metrics_lst = [self.metrics_weak, self.metrics_strong]
		for metrics in metrics_lst:
			for metric in metrics.values():
				metric.reset()
