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


class MixMatchTrainerLoc(SSTrainerABC):
	def __init__(
		self,
		model: Module,
		acti_fn: Callable,
		optim: Optimizer,
		loader_train_s_augm: DataLoader,
		loader_train_u_augms: DataLoader,
		metrics_s_weak: Dict[str, Metrics],
		metrics_u_weak: Dict[str, Metrics],
		metrics_s_strong: Dict[str, Metrics],
		metrics_u_strong: Dict[str, Metrics],
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
		self.metrics_s_weak = metrics_s_weak
		self.metrics_u_weak = metrics_u_weak
		self.metrics_s_strong = metrics_s_strong
		self.metrics_u_strong = metrics_u_strong
		self.criterion = criterion
		self.writer = writer
		self.mixer = mixer
		self.lambda_u_rampup = lambda_u_rampup

		self.metrics_recorder = MetricsRecorder(
			"train/",
			list(self.metrics_s_weak.keys()) +
			list(self.metrics_u_weak.keys()) +
			list(self.metrics_s_strong.keys()) +
			list(self.metrics_u_strong.keys()) +
			["loss", "loss_s_weak", "loss_u_weak", "loss_s_strong", "loss_u_strong"]
		)

	def train(self, epoch: int):
		self.reset_all_metrics()
		self.metrics_recorder.reset_epoch()
		self.model.train()

		loaders_zip = ZipCycle([self.loader_train_s_augm, self.loader_train_u_augms])
		iter_train = iter(loaders_zip)

		for i, item in enumerate(iter_train):
			(s_batch_augm, s_labels_weak, s_labels_strong), u_batch_augms = item

			s_batch_augm = s_batch_augm.cuda().float()
			s_labels_weak = s_labels_weak.cuda().float()
			s_labels_strong = s_labels_strong.cuda().float()
			u_batch_augms = torch.stack(u_batch_augms).cuda().float()

			# Apply mix
			s_batch_mixed, s_labels_weak_mixed, s_labels_strong_mixed, u_batch_mixed, u_labels_weak_mixed, u_labels_strong_mixed = self.mixer(
				s_batch_augm, s_labels_weak, s_labels_strong, u_batch_augms
			)

			# Compute logits
			s_logits_weak_mixed, s_logits_strong_mixed = self.model(s_batch_mixed)
			u_logits_weak_mixed, u_logits_strong_mixed = self.model(u_batch_mixed)

			s_pred_weak_mixed = self.acti_fn(s_logits_weak_mixed, dim=1)
			u_pred_weak_mixed = self.acti_fn(u_logits_weak_mixed, dim=1)
			s_pred_strong_mixed = self.acti_fn(s_logits_strong_mixed, dim=1)
			u_pred_strong_mixed = self.acti_fn(u_logits_strong_mixed, dim=1)

			# Update model
			loss, loss_s_weak, loss_u_weak, loss_s_strong, loss_u_strong = self.criterion(
				s_pred_weak_mixed, s_labels_weak_mixed, s_pred_strong_mixed, s_labels_strong_mixed,
				u_pred_weak_mixed, u_labels_weak_mixed, u_pred_strong_mixed, u_labels_strong_mixed
			)
			self.optim.zero_grad()
			loss.backward()
			self.optim.step()

			# Compute metrics
			with torch.no_grad():
				self.criterion.lambda_u = self.lambda_u_rampup.value()
				self.lambda_u_rampup.step()

				self.metrics_recorder.add_value("loss", loss.item())
				self.metrics_recorder.add_value("loss_s_weak", loss_s_weak.item())
				self.metrics_recorder.add_value("loss_u_weak", loss_u_weak.item())
				self.metrics_recorder.add_value("loss_s_strong", loss_s_strong.item())
				self.metrics_recorder.add_value("loss_u_strong", loss_u_strong.item())

				metrics_preds_labels = [
					(self.metrics_s_weak, s_pred_weak_mixed, s_labels_weak_mixed),
					(self.metrics_u_weak, u_pred_weak_mixed, u_labels_weak_mixed),
					(self.metrics_s_strong, s_pred_strong_mixed, s_labels_strong_mixed),
					(self.metrics_u_strong, u_pred_strong_mixed, u_labels_strong_mixed),
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
		return [self.metrics_s_weak, self.metrics_u_weak]
