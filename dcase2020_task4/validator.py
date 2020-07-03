import torch

from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict, List, Optional

from metric_utils.metrics import Metrics

from dcase2020_task4.metrics_recorder import MetricsRecorder, MetricsRecorderABC
from dcase2020_task4.util.checkpoint import CheckPoint
from dcase2020_task4.validator_abc import ValidatorABC


class DefaultValidator(ValidatorABC):
	def __init__(
		self,
		model: Module,
		acti_fn: Callable,
		loader: DataLoader,
		metrics: Dict[str, Metrics],
		writer: SummaryWriter,
		checkpoint: Optional[CheckPoint] = None,
		checkpoint_metric_key: Optional[str] = None,
	):
		self.model = model
		self.acti_fn = acti_fn
		self.loader = loader
		self.metrics = metrics
		self.writer = writer
		self.checkpoint = checkpoint
		self.checkpoint_metric_key = checkpoint_metric_key

		self.metrics_recorder = MetricsRecorder(
			"val/",
			list(self.metrics.keys())
		)

		if (checkpoint is None) != (checkpoint_metric_key is None):
			raise RuntimeError("If checkpoint is provided, a metric name must be used for saving best model.")

	def val(self, epoch: int):
		with torch.no_grad():
			self.reset_all_metrics()
			self.metrics_recorder.reset_epoch()
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
					self.metrics_recorder.apply_metrics_and_add(metrics_preds_labels)
					self.metrics_recorder.print_metrics(epoch, i, len(self.loader))

			print("\n")

			if self.checkpoint is not None:
				checkpoint_metric_mean = self.metrics_recorder.get_mean_epoch(self.checkpoint_metric_key)
				self.checkpoint.step(checkpoint_metric_mean)

			self.metrics_recorder.update_min_max()
			if self.writer is not None:
				self.metrics_recorder.store_in_writer(self.writer, epoch)

				for name in self.metrics.keys():
					self.writer.add_scalar("val_min/%s" % name, self.metrics_recorder.get_min(name), epoch)
					self.writer.add_scalar("val_max/%s" % name, self.metrics_recorder.get_max(name), epoch)

	def nb_examples(self) -> int:
		return len(self.loader) * self.loader.batch_size

	def get_all_metrics(self) -> List[Dict[str, Metrics]]:
		return [self.metrics]

	def get_metrics_recorder(self) -> MetricsRecorderABC:
		return self.metrics_recorder
