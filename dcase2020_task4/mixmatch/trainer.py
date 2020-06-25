import torch

from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict, List, Optional

from metric_utils.metrics import Metrics

from dcase2020_task4.util.rampup import RampUp
from dcase2020_task4.trainer_abc import SSTrainerABC
from dcase2020_task4.util.zip_cycle import ZipCycle
from dcase2020_task4.util.utils_match import get_lr
from dcase2020_task4.metrics_recorder import MetricsRecorder


class MixMatchTrainer(SSTrainerABC):
	def __init__(
		self,
		model: Module,
		acti_fn: Callable,
		optim: Optimizer,
		loader_train_s_augm: DataLoader,
		loader_train_u_augms: DataLoader,
		metrics_s: Dict[str, Metrics],
		metrics_u: Dict[str, Metrics],
		criterion: Callable,
		writer: Optional[SummaryWriter],
		mixer: Callable,
		lambda_u_rampup: RampUp
	):
		self.model = model
		self.acti_fn = acti_fn
		self.optim = optim
		self.loader_train_s_augm = loader_train_s_augm
		self.loader_train_u_augms = loader_train_u_augms
		self.metrics_s = metrics_s
		self.metrics_u = metrics_u
		self.criterion = criterion
		self.writer = writer
		self.mixer = mixer
		self.lambda_u_rampup = lambda_u_rampup

		self.metrics_recorder = MetricsRecorder(
			"train/",
			list(self.metrics_s.keys()) +
			list(self.metrics_u.keys()) +
			["loss", "loss_s", "loss_u"]
		)

	def train(self, epoch: int):
		self.reset_all_metrics()
		self.metrics_recorder.reset_epoch()
		self.model.train()

		loaders_zip = ZipCycle([self.loader_train_s_augm, self.loader_train_u_augms])
		iter_train = iter(loaders_zip)

		for i, item in enumerate(iter_train):
			(s_batch_augm, s_labels_weak), u_batch_augms = item

			s_batch_augm = s_batch_augm.cuda().float()
			s_labels_weak = s_labels_weak.cuda().float()
			u_batch_augms = torch.stack(u_batch_augms).cuda().float()

			# Apply mix
			s_batch_mixed, s_labels_mixed, u_batch_mixed, u_labels_mixed = self.mixer(
				s_batch_augm, s_labels_weak, u_batch_augms
			)

			# Compute logits
			s_logits_mixed = self.model(s_batch_mixed)
			u_logits_mixed = self.model(u_batch_mixed)

			s_pred_mixed = self.acti_fn(s_logits_mixed, dim=1)
			u_pred_mixed = self.acti_fn(u_logits_mixed, dim=1)

			# Update model
			loss, loss_s, loss_u = self.criterion(s_pred_mixed, s_labels_mixed, u_pred_mixed, u_labels_mixed)
			self.optim.zero_grad()
			loss.backward()
			self.optim.step()

			# Compute metrics
			with torch.no_grad():
				self.criterion.lambda_u = self.lambda_u_rampup.value()
				self.lambda_u_rampup.step()

				self.metrics_recorder.add_value("loss", loss.item())
				self.metrics_recorder.add_value("loss_s", loss_s.item())
				self.metrics_recorder.add_value("loss_u", loss_u.item())

				metrics_preds_labels = [
					(self.metrics_s, s_pred_mixed, s_labels_mixed),
					(self.metrics_u, u_pred_mixed, u_labels_mixed),
				]
				self.metrics_recorder.apply_metrics(metrics_preds_labels)
				self.metrics_recorder.print_metrics(epoch, i, len(loaders_zip))

		print("")

		if self.writer is not None:
			self.writer.add_scalar("hparams/lr", get_lr(self.optim), epoch)
			self.metrics_recorder.store_in_writer(self.writer, epoch)

	def nb_examples_supervised(self) -> int:
		return len(self.loader_train_s_augm) * self.loader_train_s_augm.batch_size

	def nb_examples_unsupervised(self) -> int:
		return len(self.loader_train_u_augms) * self.loader_train_u_augms.batch_size

	def get_all_metrics(self) -> List[Dict[str, Metrics]]:
		return [self.metrics_s, self.metrics_u]
