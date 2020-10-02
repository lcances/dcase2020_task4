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
		""" Store a value of a metric named. """
		raise NotImplementedError("Abstract method")

	def add_values(self, name: str, values: List[float]):
		""" Store a values of a metric named. """

	def apply_metrics_and_add(self, metrics_preds_labels: List[Tuple[Dict[str, Metrics], Tensor, Tensor]]):
		""" Call metrics with predictions and labels and store their values. """
		raise NotImplementedError("Abstract method")

	def reset_epoch(self):
		""" Reset the values stored. Should be called before starting to iterate over a dataset. """
		raise NotImplementedError("Abstract method")

	def print_metrics(self, epoch: int, i: int, len_: int):
		""" Print current epoch metrics means stored. """
		raise NotImplementedError("Abstract method")

	def print_min_max(self):
		""" Print min and max metrics stored. """
		raise NotImplementedError("Abstract method")

	def update_min_max(self):
		"""
			Update min and max with the current mean of the epoch.
			Should be called after the validation loop but before calling get_min() or get_max().
		"""
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

	def get_mins(self) -> Dict[str, float]:
		""" Get the maxs of means of all names stored.. """
		raise NotImplementedError("Abstract method")

	def get_maxs(self) -> Dict[str, float]:
		""" Get the mins of means of all names stored.. """
		raise NotImplementedError("Abstract method")

	def get_mins_maxs(self) -> (Dict[str, float], Dict[str, float]):
		""" Returns two dictionaries containing the minimums and maximums values for all epochs done. """
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
		self.stds_max = {k: 0 for k in self.keys}

		if self.prefix != "" and self.prefix[-1] != "/":
			self.prefix += "/"

		if len(set(keys)) != len(keys):
			raise RuntimeError("Duplicate found for metrics names : %s" % " ".join(keys))

	def add_value(self, name: str, value: float):
		if name not in self.data.keys():
			if self.accept_unknown_metrics:
				self.keys.append(name)
				self.data[name] = []
			else:
				raise RuntimeError("Invalid name %s. Include name in \"keys\" when building MetricsRecorder or change "
								   "\"accept_unknown_metrics\" to True." % name)
		self.data[name].append(value)

	def add_values(self, name: str, values: List[float]):
		for value in values:
			self.add_value(name, value)

	def apply_metrics_and_add(self, metrics_preds_labels: List[Tuple[Dict[str, Metrics], Tensor, Tensor]]):
		with torch.no_grad():
			for metrics, pred, label in metrics_preds_labels:
				for metric_name, metric in metrics.items():
					_mean = metric(pred, label)
					self.add_value(metric_name, metric.value.item())

	def reset_epoch(self, print_header: bool = True):
		"""
			Method called at the beginning of an epoch.
			@param print_header: Print keys of the values stored.
		"""
		self.data = {k: [] for k in self.keys}
		self.start = time()

		if print_header:
			self._print_header()

	def print_metrics(self, epoch: int, iteration: int, nb_iterations: int):
		"""
			Print current epoch means of the values stored.
			@param epoch: Current epoch number.
			@param iteration: Current iteration number in epoch.
			@param nb_iterations: Nb iteration in the current epoch.
			@return:
		"""
		percent = int(100 * (iteration + 1) / nb_iterations)

		content = ["Epoch {:3d} - {:3d}%".format(epoch + 1, percent)]
		content += [("{:.4e}".format(self.get_mean_epoch(name)).center(KEY_MAX_LENGTH)) for name in sorted(self.data.keys())]
		content += ["{:.2f}".format(time() - self.start).center(KEY_MAX_LENGTH)]

		print("- {:s} -".format(" - ".join(content)), end="\r")

	def print_min_max(self):
		self._print_header()

		content = ["{:s}".format("Min".center(16))]
		content += [("{:.4e}".format(self.get_min(name)).center(KEY_MAX_LENGTH)) for name in sorted(self.data.keys())]
		print("- {:s} -".format(" - ".join(content)))

		content = ["{:s}".format("Max".center(16))]
		content += [("{:.4e}".format(self.get_max(name)).center(KEY_MAX_LENGTH)) for name in sorted(self.data.keys())]
		print("- {:s} -".format(" - ".join(content)))

	def update_min_max(self):
		"""
			Update min and max of all epochs. Must be called before 'store_min_max_in_writer'.
		"""
		for key in self.keys:
			if len(self.data[key]) > 0:
				mean_ = self.get_mean_epoch(key)
				if mean_ > self.maxs[key]:
					self.maxs[key] = mean_
					self.stds_max[key] = self.get_std_epoch(key)
				if mean_ < self.mins[key]:
					self.mins[key] = mean_

	def store_in_writer(self, writer: SummaryWriter, epoch: int):
		"""
			Add to writer the current mean of the epoch.
		"""
		for metric_name, values in self.data.items():
			writer.add_scalar("%s%s" % (self.prefix, metric_name), np.mean(values), epoch)

	def store_min_max_in_writer(self, writer: SummaryWriter, epoch: int):
		"""
			Add to writer the current min and max of the current and previous epochs.
		"""
		for name in self.get_keys():
			writer.add_scalar("val_min/%s" % name, self.get_min(name), epoch)
			writer.add_scalar("val_max/%s" % name, self.get_max(name), epoch)
			writer.add_scalar("val_std_max/%s" % name, self.get_std_of_max(name), epoch)

	def get_keys(self) -> List[str]:
		return self.keys

	def get_mean_epoch(self, name: str) -> float:
		return float(np.mean(self.data[name]))

	def get_std_epoch(self, name: str) -> float:
		return float(np.std(self.data[name]))

	def get_min(self, name: str) -> float:
		return self.mins[name]

	def get_max(self, name: str) -> float:
		return self.maxs[name]

	def get_std_of_max(self, name: str) -> float:
		return self.stds_max[name]

	def get_mins(self) -> Dict[str, float]:
		mins = {name: self.get_min(name) for name in sorted(self.data.keys())}
		return mins

	def get_maxs(self) -> Dict[str, float]:
		maxs = {name: self.get_max(name) for name in sorted(self.data.keys())}
		return maxs

	def get_mins_maxs(self) -> (Dict[str, float], Dict[str, float]):
		return self.get_mins(), self.get_maxs()

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


def test():
	recorder = MetricsRecorder("", ["a", "b"])
	recorder.add_value("a", 1)
	recorder.add_value("a", 3)
	recorder.add_value("b", 20)
	recorder.add_value("b", 30)

	recorder.update_min_max()
	print("a Max = ", recorder.get_max("a"))
	print("b Max = ", recorder.get_max("b"))
	if recorder.get_max("a") != 2.0:
		raise RuntimeError("Test unit error")
	if recorder.get_max("b") != 25.0:
		raise RuntimeError("Test unit error")

	recorder.reset_epoch(False)
	recorder.add_value("a", 10)
	recorder.add_value("a", 20)
	recorder.add_value("b", 20)

	recorder.update_min_max()
	print("a Max = ", recorder.get_max("a"))
	print("b Max = ", recorder.get_max("b"))
	if recorder.get_max("a") != 15.0:
		raise RuntimeError("Test unit error")
	if recorder.get_max("b") != 25.0:
		raise RuntimeError("Test unit error")

	recorder.print_min_max()
	print(recorder.get_mins_maxs())


if __name__ == "__main__":
	test()
