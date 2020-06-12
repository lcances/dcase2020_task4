import numpy as np

from time import time
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict, Optional

from metric_utils.metrics import Metrics

from dcase2020_task4.fixmatch.losses.abc import FixMatchLossABC
from dcase2020_task4.util.zip_cycle import ZipCycle
from dcase2020_task4.util.utils_match import binarize_onehot_labels, get_lr
from dcase2020_task4.trainer import SSTrainer


class FixMatchTrainer(SSTrainer):
	def __init__(
		self,
		model: Module,
		acti_fn: Callable,
		optim: Optimizer,
		loader_train_s_augm_weak: DataLoader,
		loader_train_u_augms_weak_strong: DataLoader,
		metrics_s: Dict[str, Metrics],
		metrics_u: Dict[str, Metrics],
		criterion: FixMatchLossABC,
		writer: Optional[SummaryWriter],
		mode: str = "onehot",
		threshold_multihot: float = 0.5,
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

	def train(self, epoch: int):
		train_start = time()
		self.model.train()
		self.reset_metrics()

		losses, losses_s, losses_u = [], [], []
		metric_values = {
			metric_name: [] for metric_name in (list(self.metrics_s.keys()) + list(self.metrics_u.keys()))
		}

		loaders_zip = ZipCycle([self.loader_train_s_augm_weak, self.loader_train_u_augms_weak_strong])
		iter_train = iter(loaders_zip)

		for i, item in enumerate(iter_train):
			(s_batch_augm_weak, s_labels_weak), (u_batch_augm_weak, u_batch_augm_strong) = item

			s_batch_augm_weak = s_batch_augm_weak.cuda().float()
			s_labels_weak = s_labels_weak.cuda().float()
			u_batch_augm_weak = u_batch_augm_weak.cuda().float()
			u_batch_augm_strong = u_batch_augm_strong.cuda().float()

			# Compute logits
			s_logits_augm_weak = self.model(s_batch_augm_weak)
			u_logits_augm_weak = self.model(u_batch_augm_weak)
			u_logits_augm_strong = self.model(u_batch_augm_strong)

			# Compute accuracies
			s_pred_augm_weak = self.acti_fn(s_logits_augm_weak, dim=1)
			u_pred_augm_weak = self.acti_fn(u_logits_augm_weak, dim=1)
			u_pred_augm_strong = self.acti_fn(u_logits_augm_strong, dim=1)

			# Use guess u label with prediction of weak augmentation of u
			u_pred_augm_weak.detach_()
			if self.mode == "onehot":
				u_labels_weak_guessed = binarize_onehot_labels(u_pred_augm_weak)
			elif self.mode == "multihot":
				u_labels_weak_guessed = (u_pred_augm_weak > self.threshold_multihot).float()
			else:
				raise RuntimeError("Invalid argument \"mode = %s\". Use %s." % (self.mode, " or ".join(("onehot", "multihot"))))

			# Update model
			loss, loss_s, loss_u = self.criterion(s_pred_augm_weak, s_labels_weak, u_pred_augm_weak, u_pred_augm_strong, u_labels_weak_guessed)
			self.optim.zero_grad()
			loss.backward()
			self.optim.step()

			losses.append(loss.item())
			losses_s.append(loss_s.item())
			losses_u.append(loss_u.item())
			buffer = [
				"{:s}: {:.4e}".format(name, np.mean(loss_lst))
				for name, loss_lst in {"loss": losses, "loss_s": losses_s, "loss_u": losses_u}
			]

			metric_pred_labels = [
				(self.metrics_s, s_pred_augm_weak, s_labels_weak),
				(self.metrics_u, u_pred_augm_strong, u_labels_weak_guessed),
			]
			for metrics, pred, labels in metric_pred_labels:
				for metric_name, metric in metrics.items():
					mean_s = metric(pred, labels)
					buffer.append("{:s}: {:.4e}".format(metric_name, mean_s))
					metric_values[metric_name].append(metric.value.item())

			buffer.append("took: {:.2f}s".format(time() - train_start))

			print("Epoch {:d}, {:d}% \t {:s}".format(
				epoch + 1,
				int(100 * (i + 1) / len(loaders_zip)),
				" - ".join(buffer)
			), end="\r")

		print("")

		if self.writer is not None:
			self.writer.add_scalar("train/loss", float(np.mean(losses)), epoch)
			self.writer.add_scalar("train/lr", get_lr(self.optim), epoch)
			for metric_name, values in metric_values.items():
				self.writer.add_scalar("train/%s" % metric_name, float(np.mean(values)), epoch)

	def nb_examples_supervised(self) -> int:
		return len(self.loader_train_s_augm_weak) * self.loader_train_s_augm_weak.batch_size

	def nb_examples_unsupervised(self) -> int:
		return len(self.loader_train_u_augms_weak_strong) * self.loader_train_u_augms_weak_strong.batch_size

	def reset_metrics(self):
		metrics_lst = [self.metrics_s, self.metrics_u]
		for metrics in metrics_lst:
			for metric in metrics.values():
				metric.reset()
