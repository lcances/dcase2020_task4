import torch

from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict, List, Optional

from metric_utils.metrics import Metrics
from dcase2020_task4.util.utils_match import get_lr
from dcase2020_task4.metrics_recorder import MetricsRecorder
from dcase2020_task4.trainer_abc import TrainerABC
from dcase2020_task4.util.types import IterableSized


class SupervisedTrainerLoc(TrainerABC):
	def __init__(
		self,
		model: Module,
		acti_fn: Callable,
		optim: Optimizer,
		loader: IterableSized,
		criterion: Callable,
		metrics_weak: Dict[str, Metrics],
		metrics_strong: Dict[str, Metrics],
		writer: Optional[SummaryWriter],
	):
		self.model = model
		self.acti_fn = acti_fn
		self.optim = optim
		self.loader = loader
		self.criterion = criterion
		self.metrics_weak = metrics_weak
		self.metrics_strong = metrics_strong
		self.writer = writer

		self.metrics_recorder = MetricsRecorder(
			"train/",
			list(self.metrics_weak.keys()) +
			list(self.metrics_strong.keys()) +
			["loss", "loss_weak", "loss_strong"]
		)

	def train(self, epoch: int):
		self.reset_all_metrics()
		self.metrics_recorder.reset_epoch()
		self.model.train()

		iter_val = iter(self.loader)

		for i, (batch, labels_weak, labels_strong) in enumerate(iter_val):
			batch = batch.cuda().float()
			labels_weak = labels_weak.cuda().float()
			labels_strong = labels_strong.cuda().float()

			logits_weak, logits_strong = self.model(batch)

			pred_weak = self.acti_fn(logits_weak, dim=1)
			pred_strong = self.acti_fn(logits_strong, dim=1)

			loss, loss_weak, loss_strong = self.criterion(pred_weak, labels_weak, pred_strong, labels_strong)
			self.optim.zero_grad()
			loss.backward()
			self.optim.step()

			# Compute metrics
			with torch.no_grad():
				self.metrics_recorder.add_value("loss", loss.item())
				self.metrics_recorder.add_value("loss_weak", loss_weak.item())
				self.metrics_recorder.add_value("loss_strong", loss_strong.item())

				metrics_preds_labels = [
					(self.metrics_weak, pred_weak, labels_weak),
					(self.metrics_strong, pred_strong, labels_strong),
				]
				self.metrics_recorder.apply_metrics_and_add(metrics_preds_labels)
				self.metrics_recorder.print_metrics(epoch, i, self.get_nb_iterations())

		print("")

		if self.writer is not None:
			self.writer.add_scalar("hparams/lr", get_lr(self.optim), epoch)
			self.metrics_recorder.store_in_writer(self.writer, epoch)

	def get_all_metrics(self) -> List[Dict[str, Metrics]]:
		return [self.metrics_weak, self.metrics_strong]

	def get_nb_iterations(self) -> int:
		return len(self.loader)
