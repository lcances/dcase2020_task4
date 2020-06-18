import numpy as np
import torch

from time import time
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict, Optional

from metric_utils.metrics import Metrics

from dcase2020_task4.metrics_values_buffer import MetricsValuesBuffer
from dcase2020_task4.validator_abc import ValidatorABC
from dcase2020_task4.util.checkpoint import CheckPoint


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
		self.metrics_values = MetricsValuesBuffer(list(self.metrics.keys()))

	def val(self, epoch: int):
		with torch.no_grad():
			self.model.eval()
			self.reset_metrics()
			self.metrics_values.reset()

			iter_val = iter(self.loader)

			for i, (x_batch, x_label) in enumerate(iter_val):
				x_batch = x_batch.cuda().float()
				x_label = x_label.cuda().float()

				x_logits = self.model(x_batch)
				x_pred = self.acti_fn(x_logits, dim=1)

				# Compute accuracies
				with torch.no_grad():
					metrics_preds_labels = [(self.metrics, x_pred, x_label)]
					self.metrics_values.apply_metrics(metrics_preds_labels)
					self.metrics_values.print_metrics(epoch, i, len(self.loader))

			print("")

			if self.writer is not None:
				self.metrics_values.store_in_writer(self.writer, "val", epoch)

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
		writer: Optional[SummaryWriter],
		checkpoint: Optional[CheckPoint],
		checkpoint_metric_key: str = "fscore_weak",
	):
		self.model = model
		self.acti_fn = acti_fn
		self.loader = loader
		self.metrics_weak = metrics_weak
		self.metrics_strong = metrics_strong
		self.writer = writer
		self.checkpoint = checkpoint
		self.checkpoint_metric_key = checkpoint_metric_key

		self.metrics_values = MetricsValuesBuffer(list(self.metrics_weak.keys()) + list(self.metrics_strong))

	def val(self, epoch: int):
		with torch.no_grad():
			self.reset_metrics()
			self.metrics_values.reset()

			self.model.eval()

			iter_val = iter(self.loader)

			for i, (batch, labels_weak, labels_strong) in enumerate(iter_val):
				batch = batch.cuda().float()
				labels_weak = labels_weak.cuda().float()
				labels_strong = labels_strong.cuda().float()

				logits_weak, logits_strong = self.model(batch)

				pred_weak = self.acti_fn(logits_weak, dim=1)
				pred_strong = self.acti_fn(logits_strong, dim=1)

				with torch.no_grad():
					metrics_preds_labels = [
						(self.metrics_weak, pred_weak, labels_weak),
						(self.metrics_strong, pred_strong, labels_strong),
					]
					self.metrics_values.apply_metrics(metrics_preds_labels)
					self.metrics_values.print_metrics(epoch, i, len(self.loader))

					if self.checkpoint is not None:
						self.checkpoint.step(self.metrics_values.get_mean(self.checkpoint_metric_key))

			print("")

			if self.writer is not None:
				self.metrics_values.store_in_writer(self.writer, "val", epoch)

	def nb_examples(self) -> int:
		return len(self.loader) * self.loader.batch_size

	def reset_metrics(self):
		metrics_lst = [self.metrics_weak, self.metrics_strong]
		for metrics in metrics_lst:
			for metric in metrics.values():
				metric.reset()
