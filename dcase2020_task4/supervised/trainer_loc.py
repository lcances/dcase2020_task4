import torch

from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict, List, Optional

from metric_utils.metrics import Metrics
from dcase2020_task4.util.utils_match import get_lr
from dcase2020_task4.metrics_values_buffer import MetricsValuesBuffer
from dcase2020_task4.trainer_abc import TrainerABC


class SupervisedTrainerLoc(TrainerABC):
	def __init__(
		self,
		model: Module,
		acti_fn: Callable,
		optim: Optimizer,
		loader: DataLoader,
		metrics_weak: Dict[str, Metrics],
		metrics_strong: Dict[str, Metrics],
		criterion: Callable,
		writer: Optional[SummaryWriter],
	):
		self.model = model
		self.acti_fn = acti_fn
		self.optim = optim
		self.loader = loader
		self.metrics_weak = metrics_weak
		self.metrics_strong = metrics_strong
		self.criterion = criterion
		self.writer = writer

		self.metrics_values = MetricsValuesBuffer(
			list(self.metrics_weak.keys()) +
			list(self.metrics_strong.keys()) +
			["loss", "loss_weak", "loss_strong"]
		)

	def train(self, epoch: int):
		self.reset_all_metrics()
		self.metrics_values.reset()
		self.model.train()

		iter_val = iter(self.loader)

		for i, (batch, labels_weak, labels_strong) in enumerate(iter_val):
			batch = batch.cuda().float()
			labels_weak = labels_weak.cuda().float()
			labels_strong = labels_strong.cuda().float()

			logits_weak, logits_strong = self.model(batch)

			loss_weak, loss_strong, loss = self.criterion(logits_weak, logits_strong, labels_weak, labels_strong)
			self.optim.zero_grad()
			loss.backward()
			self.optim.step()

			# Compute metrics
			with torch.no_grad():
				pred_weak = self.acti_fn(logits_weak, dim=1)
				pred_strong = self.acti_fn(logits_strong, dim=1)

				self.metrics_values.add_value("loss", loss.item())
				self.metrics_values.add_value("loss_weak", loss_weak.item())
				self.metrics_values.add_value("loss_strong", loss_strong.item())

				metrics_preds_labels = [
					(self.metrics_weak, pred_weak, labels_weak),
					(self.metrics_strong, pred_strong, labels_strong),
				]
				self.metrics_values.apply_metrics(metrics_preds_labels)
				self.metrics_values.print_metrics(epoch, i, len(self.loader))

		print("")

		if self.writer is not None:
			self.writer.add_scalar("train/lr", get_lr(self.optim), epoch)
			self.metrics_values.store_in_writer(self.writer, "train", epoch)

	def nb_examples(self) -> int:
		return len(self.loader) * self.loader.batch_size

	def get_all_metrics(self) -> List[Dict[str, Metrics]]:
		return [self.metrics_weak, self.metrics_strong]
