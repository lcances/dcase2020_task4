import numpy as np
import torch

from time import time
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Tuple

from metric_utils.metrics import Metrics


class MetricsValuesBuffer:
	def __init__(
		self, keys: List[str]
	):
		self.keys = keys
		self.values = None
		self.start = None

		self.reset()

	def reset(self):
		self.values = {k: [] for k in self.keys}
		self.start = time()

	def add_value(self, name: str, value: float):
		self.values[name].append(value)

	def apply_metrics(self, metrics_preds_labels: List[Tuple[Dict[str, Metrics], Tensor, Tensor]]):
		with torch.no_grad():
			for metrics, pred, label in metrics_preds_labels:
				for metric_name, metric in metrics.items():
					_mean = metric(pred, label)
					self.add_value(metric_name, metric.value.item())

	def print_metrics(self, epoch: int, i: int, len_: int):
		prints_buffer = [
			"{:s}: {:.4e}".format(name, np.mean(values))
			for name, values in self.values.items()
		]
		prints_buffer.append("took: {:.2f}s".format(time() - self.start))

		print("Epoch {:d}, {:d}% \t {:s}".format(
			epoch + 1,
			int(100 * (i + 1) / len_),
			" - ".join(prints_buffer)
		), end="\r")

	def store_in_writer(self, writer: SummaryWriter, prefix: str, epoch: int):
		for metric_name, values in self.values.items():
			writer.add_scalar("%s/%s" % (prefix, metric_name), np.mean(values), epoch)

	def get_mean(self, name: str) -> float:
		return float(np.mean(self.values[name]))
