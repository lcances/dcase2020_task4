import torch

from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict, List, Optional

from metric_utils.metrics import Metrics

from dcase2020_task4.metrics_values_buffer import MetricsValuesBuffer
from dcase2020_task4.validator_abc import ValidatorABC
from dcase2020_task4.util.checkpoint import CheckPoint


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
		checkpoint_metric_key: str,
	):
		self.model = model
		self.acti_fn = acti_fn
		self.loader = loader
		self.metrics_weak = metrics_weak
		self.metrics_strong = metrics_strong
		self.writer = writer
		self.checkpoint = checkpoint
		self.checkpoint_metric_key = checkpoint_metric_key

		self.metrics_values = MetricsValuesBuffer(
			"val/",
			list(self.metrics_weak.keys()) + list(self.metrics_strong)
		)

	def val(self, epoch: int):
		with torch.no_grad():
			self.reset_all_metrics()
			self.metrics_values.reset_epoch()

			self.model.eval()

			iter_val = iter(self.loader)

			for i, (batch, labels_weak, labels_strong) in enumerate(iter_val):
				batch = batch.cuda().float()
				labels_weak = labels_weak.cuda().float()
				labels_strong = labels_strong.cuda().float()

				logits_weak, logits_strong = self.model(batch)

				pred_weak = self.acti_fn(logits_weak, dim=1)
				pred_strong = self.acti_fn(logits_strong, dim=1)

				# Compute metrics
				with torch.no_grad():
					metrics_preds_labels = [
						(self.metrics_weak, pred_weak, labels_weak),
						(self.metrics_strong, pred_strong, labels_strong),
					]
					self.metrics_values.apply_metrics(metrics_preds_labels)
					self.metrics_values.print_metrics(epoch, i, len(self.loader))

			print("")

			if self.checkpoint is not None:
				self.checkpoint.step(self.metrics_values.get_mean(self.checkpoint_metric_key))

			if self.writer is not None:
				self.metrics_values.store_in_writer(self.writer, epoch)

	def nb_examples(self) -> int:
		return len(self.loader) * self.loader.batch_size

	def get_all_metrics(self) -> List[Dict[str, Metrics]]:
		return [self.metrics_weak, self.metrics_strong]
