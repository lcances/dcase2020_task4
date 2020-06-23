import torch

from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict, List

from metric_utils.metrics import Metrics

from dcase2020_task4.metrics_values_buffer import MetricsValuesBuffer
from dcase2020_task4.validator_abc import ValidatorABC


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
		self.metrics_values = MetricsValuesBuffer(
			"val/",
			list(self.metrics.keys())
		)

	def val(self, epoch: int):
		with torch.no_grad():
			self.reset_all_metrics()
			self.metrics_values.reset_epoch()

			self.model.eval()

			iter_val = iter(self.loader)

			for i, (x_batch, x_label) in enumerate(iter_val):
				x_batch = x_batch.cuda().float()
				x_label = x_label.cuda().float()

				x_logits = self.model(x_batch)
				x_pred = self.acti_fn(x_logits, dim=1)

				# Compute metrics
				with torch.no_grad():
					metrics_preds_labels = [
						(self.metrics, x_pred, x_label)
					]
					self.metrics_values.apply_metrics(metrics_preds_labels)
					self.metrics_values.print_metrics(epoch, i, len(self.loader))

			print("")

			if self.writer is not None:
				self.metrics_values.store_in_writer(self.writer, epoch)

	def nb_examples(self) -> int:
		return len(self.loader) * self.loader.batch_size

	def get_all_metrics(self) -> List[Dict[str, Metrics]]:
		return [self.metrics]
