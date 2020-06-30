import torch

from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict, List, Optional

from metric_utils.metrics import Metrics

from dcase2020_task4.util.utils_match import get_lr
from dcase2020_task4.metrics_recorder import MetricsRecorder
from dcase2020_task4.trainer_abc import TrainerABC


class SupervisedTrainer(TrainerABC):
	def __init__(
		self,
		model: Module,
		acti_fn: Callable,
		optim: Optimizer,
		loader_train_s: DataLoader,
		metrics: Dict[str, Metrics],
		criterion: Callable[[Tensor, Tensor], Tensor],
		writer: Optional[SummaryWriter],
	):
		self.model = model
		self.acti_fn = acti_fn
		self.optim = optim
		self.loader_train_s = loader_train_s
		self.metrics = metrics
		self.criterion = criterion
		self.writer = writer

		self.metrics_recorder = MetricsRecorder(
			"train/",
			list(self.metrics.keys()) + ["loss"]
		)

	def train(self, epoch: int):
		self.reset_all_metrics()
		self.metrics_recorder.reset_epoch()
		self.model.train()

		iter_train = iter(self.loader_train_s)

		for i, (x, y) in enumerate(iter_train):
			x = x.cuda().float()
			y = y.cuda().float()

			# Compute logits
			logits = self.model(x)
			pred = self.acti_fn(logits, dim=1)

			# Update model
			loss = self.criterion(pred, y).mean()
			self.optim.zero_grad()
			loss.backward()
			self.optim.step()

			# Compute metrics
			with torch.no_grad():
				self.metrics_recorder.add_value("loss", loss.item())

				metrics_preds_labels = [
					(self.metrics, pred, y),
				]
				self.metrics_recorder.apply_metrics_and_add(metrics_preds_labels)
				self.metrics_recorder.print_metrics(epoch, i, len(self.loader_train_s))

		print("")

		if self.writer is not None:
			self.writer.add_scalar("hparams/lr", get_lr(self.optim), epoch)
			self.metrics_recorder.store_in_writer(self.writer, epoch)

	def nb_examples(self) -> int:
		return len(self.loader_train_s) * self.loader_train_s.batch_size

	def get_all_metrics(self) -> List[Dict[str, Metrics]]:
		return [self.metrics]
