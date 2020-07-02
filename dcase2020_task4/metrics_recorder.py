import numpy as np
import torch

from time import time
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Tuple

from metric_utils.metrics import Metrics

# Length of a column. Should be at least 10.
KEY_MAX_LENGTH = 10


class MetricsRecorderABC:
	def add_value(self, name: str, value: float):
		""" Store a value of a metric. """
		raise NotImplementedError("Abstract method")

	def apply_metrics_and_add(self, metrics_preds_labels: List[Tuple[Dict[str, Metrics], Tensor, Tensor]]):
		""" Call metrics with predictions and labels and store their values. """
		raise NotImplementedError("Abstract method")

	def reset_epoch(self):
		""" Reset the values stored. Should be called before starting to iterate over a dataset. """
		raise NotImplementedError("Abstract method")

	def print_metrics(self, epoch: int, i: int, len_: int):
		""" Print current metrics means stored. """
		raise NotImplementedError("Abstract method")

	def store_in_writer(self, writer: SummaryWriter, epoch: int):
		""" Store current metrics means in tensorboard SummaryWriter. """
		raise NotImplementedError("Abstract method")

	def get_min(self, name: str) -> float:
		""" Get the min of means of metric called name. """
		raise NotImplementedError("Abstract method")

	def get_max(self, name: str) -> float:
		""" Get the max of means of metric called name. """
		raise NotImplementedError("Abstract method")


class MetricsRecorder(MetricsRecorderABC):
	"""
		Store metric data of 1 epoch in lists.
		Useful for trainers and validators.
	"""

	def __init__(
		self, prefix: str, keys: List[str], accept_unknown_metrics: bool = False
	):
		"""
			prefix: prefix used in tensorboard names. Example: "train/" or "val/". Can be empty.
			keys: Names of all metrics used.
			accept_unknown_metrics: If false, any value added must have a name in "keys".
		"""
		self.prefix = prefix
		self.keys = keys
		self.accept_unknown_metrics = accept_unknown_metrics

		self.data = {k: [] for k in self.keys}
		self.start = time()

		self.mins = {k: np.inf for k in self.keys}
		self.maxs = {k: -np.inf for k in self.keys}
		self.can_update_min_max = False

		if len(set(keys)) != len(keys):
			raise RuntimeError("Duplicate found for metrics names : %s" % " ".join(keys))

	def add_value(self, name: str, value: float):
		self.can_update_min_max = True
		if name not in self.data.keys():
			if self.accept_unknown_metrics:
				self.keys.append(name)
				self.data[name] = []
			else:
				raise RuntimeError("Invalid name %s. Include name in \"keys\" when building MetricsRecorder or change "
								   "\"accept_unknown_metrics\" to True." % name)
		self.data[name].append(value)

	def apply_metrics_and_add(self, metrics_preds_labels: List[Tuple[Dict[str, Metrics], Tensor, Tensor]]):
		with torch.no_grad():
			for metrics, pred, label in metrics_preds_labels:
				for metric_name, metric in metrics.items():
					_mean = metric(pred, label)
					self.add_value(metric_name, metric.value.item())

	def reset_epoch(self):
		self.data = {k: [] for k in self.keys}
		self.can_update_min_max = False
		self.start = time()
		self._print_header()

	def print_metrics(self, epoch: int, i: int, len_: int):
		percent = int(100 * (i + 1) / len_)

		content = ["Epoch {:3d} - {:3d}%".format(epoch + 1, percent)]
		content += [("{:.4e}".format(self.get_mean(name)).center(KEY_MAX_LENGTH)) for name in sorted(self.data.keys())]
		content += ["{:.2f}".format(time() - self.start).center(KEY_MAX_LENGTH)]

		print("- {:s} -".format(" - ".join(content)), end="\r")

	def store_in_writer(self, writer: SummaryWriter, epoch: int):
		for metric_name, values in self.data.items():
			writer.add_scalar("%s%s" % (self.prefix, metric_name), np.mean(values), epoch)

	def get_keys(self) -> List[str]:
		return self.keys

	def get_mean(self, name: str) -> float:
		return float(np.mean(self.data[name]))

	def get_std(self, name: str) -> float:
		return float(np.std(self.data[name]))

	def get_min(self, name: str) -> float:
		if self.can_update_min_max:
			self._update_min_max()
		return self.mins[name]

	def get_max(self, name: str) -> float:
		if self.can_update_min_max:
			self._update_min_max()
		return self.maxs[name]

	def _print_header(self):
		def filter_(name: str) -> str:
			if len(name) <= KEY_MAX_LENGTH:
				return name.center(KEY_MAX_LENGTH)
			else:
				return name[:KEY_MAX_LENGTH]

		content = ["{:s}".format(self.prefix.center(16))]
		content += [filter_(name) for name in sorted(self.keys)]
		content += ["took (s)".center(KEY_MAX_LENGTH)]

		print("\n- {:s} -".format(" - ".join(content)))

	def _update_min_max(self):
		for key in self.keys:
			if len(self.data[key]) > 0:
				mean_ = self.get_mean(key)
				if mean_ > self.maxs[key]:
					self.maxs[key] = mean_
				if mean_ < self.mins[key]:
					self.mins[key] = mean_
		self.can_update_min_max = False

	def _old_print_metrics(self, epoch: int, i: int, len_: int):
		""" Unused method. """
		
		content = ["{:s}: {:.4e}".format(name, np.mean(values)) for name, values in self.data.items()]
		content += ["took: {:.2f}s".format(time() - self.start)]

		print("Epoch {:3d}, {:3d}% \t {:s}".format(
			epoch + 1,
			int(100 * (i + 1) / len_),
			" - ".join(content)
		), end="\r")
