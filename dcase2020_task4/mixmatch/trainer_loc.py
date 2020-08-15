import torch

from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict, List, Optional

from metric_utils.metrics import Metrics

from dcase2020_task4.mixmatch.losses.abc import MixMatchLossLocABC
from dcase2020_task4.metrics_recorder import MetricsRecorder
from dcase2020_task4.trainer_abc import TrainerABC
from dcase2020_task4.util.utils_match import get_lr
from dcase2020_task4.util.types import IterableSized


class MixMatchTrainerLoc(TrainerABC):
	def __init__(
		self,
		model: Module,
		acti_fn: Callable,
		optim: Optimizer,
		loader: IterableSized,
		criterion: MixMatchLossLocABC,
		metrics_s_weak: Dict[str, Metrics],
		metrics_u_weak: Dict[str, Metrics],
		metrics_s_strong: Dict[str, Metrics],
		metrics_u_strong: Dict[str, Metrics],
		writer: Optional[SummaryWriter],
		mixer: Callable,
		sharpen_fn: Callable,
		steppables: Optional[list],
	):
		self.model = model
		self.acti_fn = acti_fn
		self.optim = optim
		self.loader = loader
		self.criterion = criterion
		self.metrics_s_weak = metrics_s_weak
		self.metrics_u_weak = metrics_u_weak
		self.metrics_s_strong = metrics_s_strong
		self.metrics_u_strong = metrics_u_strong
		self.writer = writer
		self.mixer = mixer
		self.sharpen_fn = sharpen_fn
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

		iter_train = iter(self.loader)

		for i, item in enumerate(iter_train):
			(s_batch_augm, s_labels_weak, s_labels_strong), u_batch_augms = item

			s_batch_augm = s_batch_augm.cuda().float()
			s_labels_weak = s_labels_weak.cuda().float()
			s_labels_strong = s_labels_strong.cuda().float()
			u_batch_augms = torch.stack(u_batch_augms).cuda().float()

			with torch.no_grad():
				# Compute guessed label
				nb_augms = u_batch_augms.shape[0]
				u_logits_weak_augms = torch.zeros([nb_augms] + list(s_labels_weak.size())).cuda()
				u_logits_strong_augms = torch.zeros([nb_augms] + list(s_labels_strong.size())).cuda()
				for k in range(nb_augms):
					u_logits_weak_augms[k], u_logits_strong_augms[k] = self.model(u_batch_augms[k])
				u_pred_weak_augms = self.acti_fn(u_logits_weak_augms, dim=2)
				u_pred_strong_augms = self.acti_fn(u_logits_strong_augms, dim=3)

				u_label_weak_guessed = u_pred_weak_augms.mean(dim=0)
				u_label_strong_guessed = u_pred_strong_augms.mean(dim=0)

				u_label_weak_guessed = self.sharpen_fn(u_label_weak_guessed, dim=1)
				u_label_strong_guessed = self.sharpen_fn(u_label_strong_guessed, dim=2)

				# Apply mix
				s_batch_mixed, s_labels_weak_mixed, s_labels_strong_mixed, u_batch_mixed, u_labels_weak_mixed, u_labels_strong_mixed = self.mixer(
					s_batch_augm, s_labels_weak, s_labels_strong, u_batch_augms, u_label_weak_guessed, u_label_strong_guessed
				)

			# Compute logits
			s_logits_weak_mixed, s_logits_strong_mixed = self.model(s_batch_mixed)
			u_logits_weak_mixed, u_logits_strong_mixed = self.model(u_batch_mixed)

			s_pred_weak_mixed = self.acti_fn(s_logits_weak_mixed, dim=1)
			u_pred_weak_mixed = self.acti_fn(u_logits_weak_mixed, dim=1)
			s_pred_strong_mixed = self.acti_fn(s_logits_strong_mixed, dim=2)
			u_pred_strong_mixed = self.acti_fn(u_logits_strong_mixed, dim=2)

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
				self.metrics_recorder.apply_metrics_and_add(metrics_preds_labels)
				self.metrics_recorder.print_metrics(epoch, i, self.get_nb_iterations())

				for steppable in self.steppables:
					steppable.step()

		print("")

		if self.writer is not None:
			self.writer.add_scalar("hparams/lr", get_lr(self.optim), epoch)
			self.writer.add_scalar("hparams/lambda_u", self.criterion.get_lambda_u(), epoch)
			self.metrics_recorder.store_in_writer(self.writer, epoch)

	def get_all_metrics(self) -> List[Dict[str, Metrics]]:
		return [self.metrics_s_weak, self.metrics_u_weak]

	def get_nb_iterations(self) -> int:
		return len(self.loader)
