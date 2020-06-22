import torch

from torch.nn import Module
from torch.nn.functional import one_hot
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict, List, Optional

from metric_utils.metrics import Metrics

from dcase2020_task4.util.zip_cycle import ZipCycle
from dcase2020_task4.util.utils_match import get_lr
from dcase2020_task4.trainer_abc import SSTrainerABC
from dcase2020_task4.metrics_values_buffer import MetricsValuesBuffer


class FixMatchTrainerV4(SSTrainerABC):
	""" Experimental TrainerABC used for FixMatch V4 loss with only tag part. """
	def __init__(
		self,
		model: Module,
		acti_fn: Callable,
		optim: Optimizer,
		loader_train_s_augm_weak: DataLoader,
		loader_train_u_augms_weak_strong: DataLoader,
		metrics_s: Dict[str, Metrics],
		metrics_u: Dict[str, Metrics],
		criterion: Callable,
		writer: Optional[SummaryWriter],
		mode: str,
		threshold_multihot: float,
		nb_classes: int,
	):
		self.model = model
		self.acti_fn = acti_fn
		self.optim = optim
		self.loader_train_s_augm_weak = loader_train_s_augm_weak
		self.loader_train_u_augms_weak_strong = loader_train_u_augms_weak_strong
		self.metrics_s = metrics_s
		self.metrics_u = metrics_u
		self.criterion = criterion
		self.writer = writer
		self.mode = mode
		self.threshold_multihot = threshold_multihot
		self.nb_classes = nb_classes

		self.acti_count_fn = torch.softmax
		self.metrics_values = MetricsValuesBuffer(
			list(self.metrics_s.keys()) +
			list(self.metrics_u.keys()) +
			["loss", "loss_s", "loss_u", "loss_sc", "loss_uc"]
		)

	def train(self, epoch: int):
		self.reset_all_metrics()
		self.metrics_values.reset()
		self.model.train()

		loaders_zip = ZipCycle([self.loader_train_s_augm_weak, self.loader_train_u_augms_weak_strong])
		iter_train = iter(loaders_zip)

		for i, item in enumerate(iter_train):
			(s_batch_augm_weak, s_labels), (u_batch_augm_weak, u_batch_augm_strong) = item

			s_batch_augm_weak = s_batch_augm_weak.cuda().float()
			s_labels = s_labels.cuda().float()
			u_batch_augm_weak = u_batch_augm_weak.cuda().float()
			u_batch_augm_strong = u_batch_augm_strong.cuda().float()

			s_labels_count = one_hot(s_labels.long().sum(dim=1), self.nb_classes + 1)

			# Compute logits
			s_logits_augm_weak = self.model(s_batch_augm_weak)
			u_logits_augm_strong = self.model(u_batch_augm_strong)

			s_pred_augm_weak = self.acti_fn(s_logits_augm_weak, dim=1)
			u_pred_augm_strong = self.acti_fn(u_logits_augm_strong, dim=1)

			s_logits_count_augm_weak = self.model.forward_count(s_batch_augm_weak)
			u_logits_count_augm_strong = self.model.forward_count(u_batch_augm_strong)

			s_pred_count_augm_weak = self.acti_count_fn(s_logits_count_augm_weak, dim=1)
			u_pred_count_augm_strong = self.acti_count_fn(u_logits_count_augm_strong, dim=1)

			# Use guess u label with prediction of weak augmentation of u
			with torch.no_grad():
				u_logits_augm_weak = self.model(u_batch_augm_weak)
				u_logits_count_augm_weak = self.model.forward_count(u_batch_augm_weak)

				u_pred_augm_weak = self.acti_fn(u_logits_augm_weak, dim=1)
				u_pred_count_augm_weak = self.acti_count_fn(u_logits_count_augm_weak, dim=1)

				u_labels_weak_guessed = (u_pred_augm_weak > self.threshold_multihot).float()

			# Update model
			loss, loss_s, loss_u, loss_sc, loss_uc = self.criterion(
				s_pred_augm_weak, s_labels,
				u_pred_augm_weak, u_pred_augm_strong, u_labels_weak_guessed,
				s_pred_count_augm_weak, s_labels_count,
				u_pred_count_augm_weak, u_pred_count_augm_strong,
			)
			self.optim.zero_grad()
			loss.backward()
			self.optim.step()

			# Compute metrics
			with torch.no_grad():
				self.metrics_values.add_value("loss", loss.item())
				self.metrics_values.add_value("loss_s", loss_s.item())
				self.metrics_values.add_value("loss_u", loss_u.item())
				self.metrics_values.add_value("loss_sc", loss_sc.item())
				self.metrics_values.add_value("loss_uc", loss_uc.item())

				metrics_preds_labels = [
					(self.metrics_s, s_pred_augm_weak, s_labels),
					(self.metrics_u, u_pred_augm_strong, u_labels_weak_guessed),
				]
				self.metrics_values.apply_metrics(metrics_preds_labels)
				self.metrics_values.print_metrics(epoch, i, len(loaders_zip))

			print("")

		if self.writer is not None:
			self.writer.add_scalar("train/lr", get_lr(self.optim), epoch)
			self.metrics_values.store_in_writer(self.writer, "train", epoch)

	def nb_examples_supervised(self) -> int:
		return len(self.loader_train_s_augm_weak) * self.loader_train_s_augm_weak.batch_size

	def nb_examples_unsupervised(self) -> int:
		return len(self.loader_train_u_augms_weak_strong) * self.loader_train_u_augms_weak_strong.batch_size

	def get_all_metrics(self) -> List[Dict[str, Metrics]]:
		return [self.metrics_s, self.metrics_u]
