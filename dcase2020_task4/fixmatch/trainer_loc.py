import torch

from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict, List, Optional

from metric_utils.metrics import Metrics

from dcase2020_task4.fixmatch.losses.abc import FixMatchLossLocABC
from dcase2020_task4.trainer_abc import SSTrainerABC
from dcase2020_task4.metrics_recorder import MetricsRecorder

from dcase2020_task4.util.avg_distributions import AvgDistributions
from dcase2020_task4.util.utils_match import get_lr
from dcase2020_task4.util.zip_cycle import ZipCycle


class FixMatchTrainerLoc(SSTrainerABC):
	def __init__(
		self,
		model: Module,
		acti_fn: Callable,
		optim: Optimizer,
		loader_train_s_augm_weak: DataLoader,
		loader_train_u_augms_weak_strong: DataLoader,
		metrics_s_weak: Dict[str, Metrics],
		metrics_u_weak: Dict[str, Metrics],
		metrics_s_strong: Dict[str, Metrics],
		metrics_u_strong: Dict[str, Metrics],
		criterion: FixMatchLossLocABC,
		writer: Optional[SummaryWriter],
		distributions: Optional[AvgDistributions],
		threshold_multihot: float,
		steppables: Optional[list],
	):
		self.model = model
		self.acti_fn = acti_fn
		self.optim = optim
		self.loader_train_s_augm_weak = loader_train_s_augm_weak
		self.loader_train_u_augms_weak_strong = loader_train_u_augms_weak_strong
		self.metrics_s_weak = metrics_s_weak
		self.metrics_u_weak = metrics_u_weak
		self.metrics_s_strong = metrics_s_strong
		self.metrics_u_strong = metrics_u_strong
		self.criterion = criterion
		self.writer = writer
		self.distributions = distributions
		self.threshold_multihot = threshold_multihot
		self.steppables = steppables if steppables is not None else []

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

		loaders_zip = ZipCycle([self.loader_train_s_augm_weak, self.loader_train_u_augms_weak_strong])
		iter_train = iter(loaders_zip)

		for i, item in enumerate(iter_train):
			(s_batch_augm_weak, s_labels_weak, s_labels_strong), (u_batch_augm_weak, u_batch_augm_strong) = item

			s_batch_augm_weak = s_batch_augm_weak.cuda().float()
			s_labels_weak = s_labels_weak.cuda().float()
			s_labels_strong = s_labels_strong.cuda().float()
			u_batch_augm_weak = u_batch_augm_weak.cuda().float()
			u_batch_augm_strong = u_batch_augm_strong.cuda().float()

			# Compute logits
			s_logits_weak_augm_weak, s_logits_strong_augm_weak = self.model(s_batch_augm_weak)
			u_logits_weak_augm_strong, u_logits_strong_augm_strong = self.model(u_batch_augm_strong)

			s_pred_weak_augm_weak = self.acti_fn(s_logits_weak_augm_weak, dim=1)
			u_pred_weak_augm_strong = self.acti_fn(u_logits_weak_augm_strong, dim=1)

			s_pred_strong_augm_weak = self.acti_fn(s_logits_strong_augm_weak, dim=(1, 2))
			u_pred_strong_augm_strong = self.acti_fn(u_logits_strong_augm_strong, dim=(1, 2))

			with torch.no_grad():
				u_logits_weak_augm_weak, u_logits_strong_augm_weak = self.model(u_batch_augm_weak)
				u_pred_weak_augm_weak = self.acti_fn(u_logits_weak_augm_weak, dim=1)
				u_pred_strong_augm_weak = self.acti_fn(u_logits_strong_augm_weak, dim=(1, 2))

				if self.distributions is not None:
					self.distributions.add_batch_pred(s_labels_weak, "labeled")
					self.distributions.add_batch_pred(u_pred_weak_augm_weak, "unlabeled")

					s_pred_weak_augm_weak = self.distributions.apply_distribution_alignment(s_pred_weak_augm_weak, dim=1)
					u_pred_weak_augm_weak = self.distributions.apply_distribution_alignment(u_pred_weak_augm_weak, dim=1)

				# Use guess u label with prediction of weak augmentation of u
				u_labels_weak_guessed = (u_pred_weak_augm_weak > self.threshold_multihot).float()
				u_labels_strong_guessed = (u_pred_strong_augm_weak > self.threshold_multihot).float()

			# Update model
			loss, loss_s_weak, loss_u_weak, loss_s_strong, loss_u_strong = self.criterion(
				s_pred_weak_augm_weak, s_labels_weak,
				u_pred_weak_augm_weak, u_pred_weak_augm_strong, u_labels_weak_guessed,
				s_pred_strong_augm_weak, s_labels_strong,
				u_pred_strong_augm_weak, u_pred_strong_augm_strong, u_labels_strong_guessed,
			)
			self.optim.zero_grad()
			loss.backward()
			self.optim.step()

			# Compute metrics
			with torch.no_grad():
				self.metrics_recorder.add_value("loss", loss.item())
				self.metrics_recorder.add_value("loss_s_weak", loss_s_weak.item())
				self.metrics_recorder.add_value("loss_u_weak", loss_u_weak.item())
				self.metrics_recorder.add_value("loss_s_strong", loss_s_strong.item())
				self.metrics_recorder.add_value("loss_u_strong", loss_u_strong.item())

				metrics_preds_labels = [
					(self.metrics_s_weak, s_pred_weak_augm_weak, s_labels_weak),
					(self.metrics_u_weak, u_pred_weak_augm_strong, u_labels_weak_guessed),
					(self.metrics_s_strong, s_pred_strong_augm_weak, s_labels_strong),
					(self.metrics_u_strong, u_pred_strong_augm_strong, u_labels_strong_guessed),
				]
				self.metrics_recorder.apply_metrics_and_add(metrics_preds_labels)
				self.metrics_recorder.print_metrics(epoch, i, len(loaders_zip))

				for steppable in self.steppables:
					steppable.step()

		print("")

		if self.writer is not None:
			self.writer.add_scalar("hparams/lr", get_lr(self.optim), epoch)
			self.writer.add_scalar("hparams/lambda_u", self.criterion.get_lambda_u(), epoch)
			self.metrics_recorder.store_in_writer(self.writer, epoch)

	def nb_examples_supervised(self) -> int:
		return len(self.loader_train_s_augm_weak) * self.loader_train_s_augm_weak.batch_size

	def nb_examples_unsupervised(self) -> int:
		return len(self.loader_train_u_augms_weak_strong) * self.loader_train_u_augms_weak_strong.batch_size

	def get_all_metrics(self) -> List[Dict[str, Metrics]]:
		return [self.metrics_s_weak, self.metrics_u_weak, self.metrics_s_strong, self.metrics_u_strong]
