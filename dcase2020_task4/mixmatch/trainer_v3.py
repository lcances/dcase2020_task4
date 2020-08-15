import torch

from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict, List, Optional

from metric_utils.metrics import Metrics

from dcase2020_task4.metrics_recorder import MetricsRecorder
from dcase2020_task4.mixmatch.losses.abc import MixMatchLossTagABC
from dcase2020_task4.trainer_abc import TrainerABC
from dcase2020_task4.util.guessers.abc import GuesserModelABC
from dcase2020_task4.util.utils_match import get_lr
from dcase2020_task4.util.types import IterableSized


class MixMatchTrainerV3(TrainerABC):
	def __init__(
		self,
		model: Module,
		acti_fn: Callable,
		optim: Optimizer,
		loader: IterableSized,
		criterion: MixMatchLossTagABC,
		guesser: GuesserModelABC,
		metrics_s: Dict[str, Metrics],
		metrics_u: Dict[str, Metrics],
		writer: Optional[SummaryWriter],
		mixer: Callable,
		steppables: Optional[list],
	):
		self.model = model
		self.acti_fn = acti_fn
		self.optim = optim
		self.loader = loader
		self.criterion = criterion
		self.guesser = guesser
		self.metrics_s = metrics_s
		self.metrics_u = metrics_u
		self.writer = writer
		self.mixer = mixer
		self.steppables = steppables if steppables is not None else []

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

		iter_train = iter(self.loader)

		for i, item in enumerate(iter_train):
			(s_batch_augm, s_labels), (u_batch_augms, u_batch) = item

			s_batch_augm = s_batch_augm.cuda().float()
			s_labels = s_labels.cuda().float()
			u_batch_augms = torch.stack(u_batch_augms).cuda().float()
			u_batch = u_batch.cuda().float()

			with torch.no_grad():
				u_label_guessed = self.guesser.guess_label(u_batch_augms, dim=1)

				# Apply mix
				s_batch_mixed, s_labels_mixed, u_batch_mixed, u_labels_mixed = self.mixer(
					s_batch_augm, s_labels, u_batch.unsqueeze(dim=0), u_label_guessed
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
				self.metrics_recorder.add_value("loss", loss.item())
				self.metrics_recorder.add_value("loss_s", loss_s.item())
				self.metrics_recorder.add_value("loss_u", loss_u.item())

				metrics_preds_labels = [
					(self.metrics_s, s_pred_mixed, s_labels_mixed),
					(self.metrics_u, u_pred_mixed, u_labels_mixed),
				]
				self.metrics_recorder.apply_metrics_and_add(metrics_preds_labels)
				self.metrics_recorder.print_metrics(epoch, i, self.get_nb_iterations())

				for steppable in self.steppables:
					steppable.step()

		print("")

		if self.writer is not None:
			self.writer.add_scalar("hparams/lr", get_lr(self.optim), epoch)
			self.writer.add_scalar("hparams/lambda_s", self.criterion.get_lambda_s(), epoch)
			self.writer.add_scalar("hparams/lambda_u", self.criterion.get_lambda_u(), epoch)
			self.metrics_recorder.store_in_writer(self.writer, epoch)

	def get_all_metrics(self) -> List[Dict[str, Metrics]]:
		return [self.metrics_s, self.metrics_u]

	def get_nb_iterations(self) -> int:
		return len(self.loader)
