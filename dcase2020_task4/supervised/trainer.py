import torch

from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict, List, Optional

from metric_utils.metrics import Metrics

from dcase2020_task4.util.utils_match import get_lr
from dcase2020_task4.metrics_recorder import MetricsRecorder
from dcase2020_task4.trainer_abc import TrainerABC
from dcase2020_task4.util.types import IterableSized


class SupervisedTrainer(TrainerABC):
	def __init__(
		self,
		model: Module,
		acti_fn: Callable,
		optim: Optimizer,
		loader: IterableSized,
		criterion: Callable[[Tensor, Tensor], Tensor],
		metrics: Dict[str, Metrics],
		writer: Optional[SummaryWriter],
	):
		self.model = model
		self.acti_fn = acti_fn
		self.optim = optim
		self.loader = loader
		self.criterion = criterion
		self.metrics = metrics
		self.writer = writer

		self.metrics_recorder = MetricsRecorder(
			"train/",
			list(self.metrics.keys()) + ["loss"]
		)

	def train(self, epoch: int):
		self.reset_all_metrics()
		self.metrics_recorder.reset_epoch()
		self.model.train()

		iter_train = iter(self.loader)

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
				self.metrics_recorder.print_metrics(epoch, i, self.get_nb_iterations())

		print("")

		if self.writer is not None:
			self.writer.add_scalar("hparams/lr", get_lr(self.optim), epoch)
			self.metrics_recorder.store_in_writer(self.writer, epoch)

	def get_all_metrics(self) -> List[Dict[str, Metrics]]:
		return [self.metrics]

	def get_nb_iterations(self) -> int:
		return len(self.loader)
