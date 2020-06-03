import numpy as np
import torch

from abc import ABC
from time import time
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict

from dcase2020.pytorch_metrics.metrics import Metrics


class Validator(ABC):
	def val(self, epoch: int):
		raise NotImplementedError("Abstract method")

	def nb_examples(self) -> int:
		raise NotImplementedError("Abstract method")


class DefaultValidator(Validator):
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
			for metric in self.metrics.values():
				metric.reset()
			self.model.eval()

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
