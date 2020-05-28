import numpy as np

from easydict import EasyDict as edict
from time import time
from torch.nn import Module
from torch.nn.functional import one_hot
from torch.optim import SGD
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, List

from dcase2020.pytorch_metrics.metrics import Metrics
from dcase2020_task4.learner import DefaultLearner
from dcase2020_task4.util.utils_match import get_lr, build_writer, cross_entropy, cross_entropy_with_logits
from dcase2020_task4.trainer import Trainer
from dcase2020_task4.validate import DefaultValidator


class SupervisedTrainer(Trainer):
	def __init__(self, model: Module, acti_fn: Callable, optim: Optimizer, loader: DataLoader, criterion: Callable,
				 metrics: Metrics, writer: SummaryWriter, hparams: edict):
		self.model = model
		self.acti_fn = acti_fn
		self.optim = optim
		self.loader = loader
		self.criterion = criterion
		self.metrics = metrics
		self.writer = writer
		self.nb_classes = hparams.nb_classes

	def train(self, epoch: int):
		train_start = time()
		self.metrics.reset()
		self.model.train()

		losses, acc_train = [], []
		iter_train = iter(self.loader)

		for i, (x, y) in enumerate(iter_train):
			x, y_num = x.cuda().float(), y.cuda().long()
			y = one_hot(y_num, self.nb_classes)

			# Compute logits
			logits = self.model(x)

			# Compute accuracy
			pred = self.acti_fn(logits)
			accuracy = self.metrics(pred, y)

			# Update model
			loss = self.criterion(logits, y).mean()
			self.optim.zero_grad()
			loss.backward()
			self.optim.step()

			# Store data
			losses.append(loss.item())
			acc_train.append(self.metrics.value.item())

			# logs
			print("Epoch {}, {:d}% \t loss: {:.4e} - acc: {:.4e} - took {:.2f}s".format(
				epoch + 1,
				int(100 * (i + 1) / len(self.loader)),
				loss.item(),
				accuracy,
				time() - train_start
			), end="\r")

		print("")

		self.writer.add_scalar("train/loss", float(np.mean(losses)), epoch)
		self.writer.add_scalar("train/acc", float(np.mean(acc_train)), epoch)
		self.writer.add_scalar("train/lr", get_lr(self.optim), epoch)

	def nb_examples(self) -> int:
		return len(self.loader) * self.loader.batch_size


def train_supervised(
	model: Module,
	acti_fn: Callable,
	loader_train_full: DataLoader,
	loader_val: DataLoader,
	metrics_s: Metrics,
	metrics_val_lst: List[Metrics],
	metrics_names: List[str],
	hparams: edict,
	suffix: str
):
	hparams.lr = 1e-2
	hparams.weight_decay = 1e-4

	optim = SGD(model.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay)

	hparams.train_name = "Supervised"
	writer = build_writer(hparams, suffix=suffix)

	trainer = SupervisedTrainer(
		model, acti_fn, optim, loader_train_full, cross_entropy_with_logits, metrics_s, writer, hparams
	)
	validator = DefaultValidator(
		model, acti_fn, loader_val, cross_entropy, metrics_val_lst, metrics_names, writer, hparams.nb_classes
	)
	learner = DefaultLearner(trainer, validator, hparams.nb_epochs)

	print("\nStart %s training (%d epochs, %d train examples supervised, %d valid examples)..." % (
		hparams.train_name,
		hparams.nb_epochs,
		trainer.nb_examples(),
		validator.nb_examples()
	))
	start = time()
	learner.start()
	print("End %s training. (duration = %.2f)" % (hparams.train_name, time() - start))

	writer.add_hparams(hparam_dict=dict(hparams), metric_dict={})
	writer.close()
