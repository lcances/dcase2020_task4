import numpy as np
import torch

from time import time
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Tuple

from metric_utils.metrics import Metrics

# Length max of keys names. Should be at least 10.
KEY_MAX_LENGTH = 10


class MetricsValuesBuffer:
	"""
		Store metric data of 1 epoch in lists.
		Useful for trainers and validators.
	"""

	def __init__(
		self, prefix: str, keys: List[str]
	):
		self.prefix = prefix
		self.keys = keys
		self.data = {k: [] for k in self.keys}
		self.start = time()

		if len(set(keys)) != len(keys):
			raise RuntimeError("Duplicate found for metrics names : %s" % " ".join(keys))

	def add_value(self, name: str, value: float):
		self.data[name].append(value)

	def apply_metrics(self, metrics_preds_labels: List[Tuple[Dict[str, Metrics], Tensor, Tensor]]):
		with torch.no_grad():
			for metrics, pred, label in metrics_preds_labels:
				for metric_name, metric in metrics.items():
					_mean = metric(pred, label)
					self.add_value(metric_name, metric.value.item())

	def reset_epoch(self):
		self.data = {k: [] for k in self.keys}
		self.start = time()

		def filter(name: str) -> str:
			if len(name) <= KEY_MAX_LENGTH:
				return name.center(KEY_MAX_LENGTH)
			else:
				return name[:KEY_MAX_LENGTH]

		content = ["{:s}".format(self.prefix.center(16))]
		content += [filter(name) for name in self.keys]
		content += ["took (s)".center(KEY_MAX_LENGTH)]

		print("| {:s} |".format(" | ".join(content)))

	def print_metrics(self, epoch: int, i: int, len_: int):
		percent = int(100 * (i + 1) / len_)

		content = ["Epoch {:3d} | {:3d}%".format(epoch, percent)]
		content += [("{:.4e}".format(self.get_mean(name)).center(KEY_MAX_LENGTH)) for name in self.data.keys()]
		content += ["{:.2f}".format(time() - self.start).center(KEY_MAX_LENGTH)]

		print("| {:s} |".format(" | ".join(content)), end="\r")

	def print_metrics_old(self, epoch: int, i: int, len_: int):
		prints_buffer = [
			"{:s}: {:.4e}".format(name, np.mean(values))
			for name, values in self.data.items()
		]
		prints_buffer.append("took: {:.2f}s".format(time() - self.start))

		print("Epoch {:3d}, {:3d}% \t {:s}".format(
			epoch + 1,
			int(100 * (i + 1) / len_),
			" - ".join(prints_buffer)
		), end="\r")

	def store_in_writer(self, writer: SummaryWriter, epoch: int):
		for metric_name, values in self.data.items():
			writer.add_scalar("%s%s" % (self.prefix, metric_name), np.mean(values), epoch)

	def get_mean(self, name: str) -> float:
		return float(np.mean(self.data[name]))

	def get_std(self, name: str) -> float:
		return float(np.std(self.data[name]))
