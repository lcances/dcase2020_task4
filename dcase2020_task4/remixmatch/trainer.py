import torch

from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict, List, Optional

from metric_utils.metrics import Metrics

from dcase2020_task4.metrics_recorder import MetricsRecorder
from dcase2020_task4.remixmatch.losses.abc import ReMixMatchLossTagABC
from dcase2020_task4.remixmatch.self_label import SelfSupervisedABC
from dcase2020_task4.trainer_abc import TrainerABC

from dcase2020_task4.util.avg_distributions import AvgDistributions
from dcase2020_task4.util.guessers.abc import GuesserModelABC
from dcase2020_task4.util.types import IterableSized
from dcase2020_task4.util.utils_match import get_lr


class ReMixMatchTrainer(TrainerABC):
	def __init__(
		self,
		model: Module,
		acti_fn: Callable,
		acti_rot_fn: Callable,
		optim: Optimizer,
		loader: IterableSized,
		criterion: ReMixMatchLossTagABC,
		guesser: GuesserModelABC,
		metrics_s: Dict[str, Metrics],
		metrics_u: Dict[str, Metrics],
		metrics_u1: Dict[str, Metrics],
		metrics_r: Dict[str, Metrics],
		writer: Optional[SummaryWriter],
		mixer: Callable,
		distributions: Optional[AvgDistributions],
		ss_transform: Optional[SelfSupervisedABC],
		steppables: Optional[list],
	):
		"""
			Note: model must implements torch.nn.Module and implements a method "forward_rot".
		"""
		self.model = model
		self.acti_fn = acti_fn
		self.optim = optim
		self.loader = loader
		self.criterion = criterion
		self.guesser = guesser
		self.metrics_s = metrics_s
		self.metrics_u = metrics_u
		self.metrics_u1 = metrics_u1
		self.metrics_r = metrics_r
		self.writer = writer
		self.mixer = mixer
		self.distributions = distributions
		self.ss_transform = ss_transform
		self.steppables = steppables if steppables is not None else []

		self.acti_rot_fn = acti_rot_fn
		self.metrics_recorder = MetricsRecorder(
			"train/",
			list(self.metrics_s.keys()) +
			list(self.metrics_u.keys()) +
			list(self.metrics_u1.keys()) +
			list(self.metrics_r.keys()) +
			["loss", "loss_s", "loss_u", "loss_u1", "loss_r"]
		)

		if not hasattr(model, "forward_rot"):
			raise RuntimeError("Model must implements a method \"forward_rot\" for compute rotation loss.")

	def train(self, epoch: int):
		self.reset_all_metrics()
		self.metrics_recorder.reset_epoch()
		self.model.train()

		iter_train = iter(self.loader)

		for i, item in enumerate(iter_train):
			(s_batch_augm_strong, s_labels), (u_batch_augm_weak, u_batch_augm_strongs) = item

			s_batch_augm_strong = s_batch_augm_strong.cuda().float()
			s_labels = s_labels.cuda().float()
			u_batch_augm_weak = u_batch_augm_weak.cuda().float()
			u_batch_augm_strongs = torch.stack(u_batch_augm_strongs).cuda().float()

			with torch.no_grad():
				u_label_guessed = self.guesser.guess_label(u_batch_augm_weak, dim=1)
				u_pred_augm_weak = self.guesser.get_last_pred()

				if self.distributions is not None:
					self.distributions.add_batch_pred(s_labels, "labeled")
					self.distributions.add_batch_pred(u_pred_augm_weak, "unlabeled")

				# Get strongly augmented batch "batch_u1"
				u1_batch = u_batch_augm_strongs[0, :].clone()
				u1_label = u_label_guessed.clone()

				# Apply mix
				s_batch_mixed, s_labels_mixed, u_batch_mixed, u_labels_mixed = \
					self.mixer(s_batch_augm_strong, s_labels, u_batch_augm_weak, u_batch_augm_strongs, u_label_guessed)

				if self.ss_transform is not None:
					u1_batch_self_super, u1_label_self_super = self.ss_transform.create_batch_label(u1_batch)
				else:
					u1_batch_self_super, u1_label_self_super = None, None

			# Predict labels for x (mixed), u (mixed) and u1 (strong augment)
			s_logits_mixed = self.model(s_batch_mixed)
			u_logits_mixed = self.model(u_batch_mixed)
			u1_logits = self.model(u1_batch)

			s_pred_mixed = self.acti_fn(s_logits_mixed, dim=1)
			u_pred_mixed = self.acti_fn(u_logits_mixed, dim=1)
			u1_pred = self.acti_fn(u1_logits, dim=1)

			if u1_batch_self_super is not None:
				# Predict rotation for strong augment u1
				u1_logits_self_super = self.model.forward_rot(u1_batch_self_super)
				u1_pred_self_super = self.acti_rot_fn(u1_logits_self_super, dim=1)
			else:
				u1_pred_self_super = None

			# Update model
			loss, loss_s, loss_u, loss_u1, loss_r = self.criterion(
				s_pred_mixed, s_labels_mixed,
				u_pred_mixed, u_labels_mixed,
				u1_pred, u1_label,
				u1_pred_self_super, u1_label_self_super
			)
			self.optim.zero_grad()
			loss.backward()
			self.optim.step()

			# Compute metrics
			with torch.no_grad():
				self.metrics_recorder.add_value("loss", loss.item())
				self.metrics_recorder.add_value("loss_s", loss_s.item())
				self.metrics_recorder.add_value("loss_u", loss_u.item())
				self.metrics_recorder.add_value("loss_u1", loss_u1.item())
				self.metrics_recorder.add_value("loss_r", loss_r.item())

				metrics_preds_labels = [
					(self.metrics_s, s_pred_mixed, s_labels_mixed),
					(self.metrics_u, u_pred_mixed, u_labels_mixed),
					(self.metrics_u1, u1_pred, u1_label),
				]
				if u1_pred_self_super is not None and u1_label_self_super is not None:
					metrics_preds_labels += [
						(self.metrics_r, u1_pred_self_super, u1_label_self_super),
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
			self.writer.add_scalar("hparams/lambda_u1", self.criterion.get_lambda_u1(), epoch)
			self.writer.add_scalar("hparams/lambda_r", self.criterion.get_lambda_r(), epoch)
			self.metrics_recorder.store_in_writer(self.writer, epoch)

	def get_all_metrics(self) -> List[Dict[str, Metrics]]:
		return [self.metrics_s, self.metrics_u, self.metrics_u1, self.metrics_r]

	def get_nb_iterations(self) -> int:
		return len(self.loader)
