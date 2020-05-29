import numpy as np
import torch

from abc import ABC
from time import time
from torch.nn import Module
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Optional, List

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
		metrics_lst: List[Metrics],
		metrics_names: List[str],
		writer: SummaryWriter,
		nb_classes: int,
		mode: str,
	):
		self.model = model
		self.acti_fn = acti_fn
		self.loader = loader
		self.metrics_lst = metrics_lst
		self.metrics_names = metrics_names
		self.writer = writer
		self.nb_classes = nb_classes

		self.pre_batch_fn = lambda batch: batch.cuda().float()
		if mode == "onehot":
			self.pre_label_fn = lambda label: one_hot(label.cuda().long(), self.nb_classes).float()
		elif mode == "multihot":
			self.pre_label_fn = lambda label: label.cuda().float()
		else:
			raise RuntimeError("Invalid argument \"mode = %s\". Use %s." % (mode, " or ".join(("onehot", "multihot"))))

	def val(self, epoch: int):
		with torch.no_grad():
			val_start = time()
			for metrics in self.metrics_lst:
				metrics.reset()
			self.model.eval()

			metrics_values = [[] for _ in range(len(self.metrics_lst))]
			iter_val = iter(self.loader)
			for i, (x, y) in enumerate(iter_val):
				x = self.pre_batch_fn(x)
				y = self.pre_label_fn(y)

				logits_x = self.model(x)
				pred_x = self.acti_fn(logits_x)

				buffer = []

				# Compute metrics and store them in buffer for print and in metrics_values for writer
				for values, metrics, name in zip(metrics_values, self.metrics_lst, self.metrics_names):
					cur_mean_value = metrics(pred_x, y)
					buffer.append("%s: %.4e" % (name, cur_mean_value))
					values.append(metrics.value.item())

				# Add time elapsed since the beginning of validation
				buffer.append("took %.2fs" % (time() - val_start))

				# Print buffer
				print("Epoch {}, {:d}% \t {:s}".format(
					epoch + 1,
					int(100 * (i + 1) / len(self.loader)),
					" - ".join(buffer)
				), end="\r")

			print("")

			for values, name in zip(metrics_values, self.metrics_names):
				self.writer.add_scalar("val/%s" % name, float(np.mean(values)), epoch)

	def nb_examples(self) -> int:
		return len(self.loader) * self.loader.batch_size
