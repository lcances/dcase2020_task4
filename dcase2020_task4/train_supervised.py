import numpy as np
import os.path as osp

from easydict import EasyDict as edict
from time import time
from torch.nn import Module, CrossEntropyLoss
from torch.nn.functional import one_hot
from torch.optim import SGD
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable

from dcase2020.pytorch_metrics.metrics import Metrics
from dcase2020_task4.util.confidence_acc import CategoricalConfidenceAccuracy
from dcase2020_task4.util.utils_match import get_lr
from dcase2020_task4.validate import val


def test_supervised(
	model: Module, acti_fn: Callable, loader_train: DataLoader, loader_val: DataLoader, hparams: edict, suffix: str = ""
):
	hparams.lr = 1e-2
	hparams.weight_decay = 1e-4

	hparams.train_name = "Supervised"
	dirname = "%s_%s_%s_%s" % (hparams.train_name, hparams.model_name, suffix, hparams.begin_date)
	dirpath = osp.join(hparams.logdir, dirname)
	writer = SummaryWriter(log_dir=dirpath, comment=hparams.train_name)

	optim = SGD(model.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay)
	metrics_s = CategoricalConfidenceAccuracy(hparams.confidence)
	metrics_val = CategoricalConfidenceAccuracy(hparams.confidence)
	criterion = CrossEntropyLoss()

	start = time()
	print("\nStart %s training (%d epochs, %d train examples, %d valid examples)..." % (
		hparams.train_name,
		hparams.nb_epochs,
		len(loader_train) * loader_train.batch_size,
		len(loader_val) * loader_val.batch_size
	))

	for e in range(hparams.nb_epochs):
		losses, acc_train = train_supervised(
			model, acti_fn, optim, loader_train, hparams.nb_classes, criterion, metrics_s, e
		)
		acc_val, acc_maxs = val(
			model, acti_fn, loader_val, hparams.nb_classes, metrics_val, e
		)
		maxes = 0
		maxes += 1

		writer.add_scalar("train/loss", float(np.mean(losses)), e)
		writer.add_scalar("train/acc", float(np.mean(acc_train)), e)
		writer.add_scalar("train/lr", get_lr(optim), e)
		writer.add_scalar("val/acc", float(np.mean(acc_val)), e)
		writer.add_scalar("val/maxs", float(np.mean(acc_maxs)), e)

	print("End Supervised training. (duration = %.2f)" % (time() - start))

	writer.add_hparams(hparam_dict=dict(hparams), metric_dict={})
	writer.close()


def train_supervised(
	model: Module,
	acti_fn: Callable,
	optimizer: Optimizer,
	loader: DataLoader,
	nb_classes: int,
	criterion: Callable,
	metrics: Metrics,
	epoch: int
) -> (list, list):
	train_start = time()
	metrics.reset()
	model.train()

	losses, accuracies = [], []
	iter_train = iter(loader)
	for i, (x, y) in enumerate(iter_train):
		x, y_num = x.cuda().float(), y.cuda().long()
		y = one_hot(y_num, nb_classes)

		# Compute logits
		logits = model(x)

		# Compute accuracy
		pred = acti_fn(logits)
		accuracy = metrics(pred, y)

		# Update model
		loss = criterion(logits, y_num)  # note softmax is applied inside CrossEntropy
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# Store data
		losses.append(loss.item())
		accuracies.append(metrics.value.item())

		# logs
		print("Epoch {}, {:d}% \t loss: {:.4e} - acc: {:.4e} - took {:.2f}s".format(
			epoch + 1,
			int(100 * (i + 1) / len(loader)),
			loss.item(),
			accuracy,
			time() - train_start
		), end="\r")

	print("")
	return losses, accuracies
