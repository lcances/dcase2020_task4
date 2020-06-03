
from easydict import EasyDict as edict
from torch.nn import Module
from torch.optim import SGD
from torch.nn.functional import binary_cross_entropy
from torch.utils.data import DataLoader
from typing import Callable, Dict

from dcase2020.pytorch_metrics.metrics import Metrics
from dcase2020_task4.learner import DefaultLearner
from dcase2020_task4.supervised.trainer import SupervisedTrainer
from dcase2020_task4.util.utils_match import build_writer, cross_entropy
from dcase2020_task4.validator import DefaultValidator


def train_supervised(
	model: Module,
	acti_fn: Callable,
	loader_train_full: DataLoader,
	loader_val: DataLoader,
	metric_s: Metrics,
	metrics_val: Dict[str, Metrics],
	hparams: edict,
	suffix: str
):
	optim = SGD(model.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay)

	hparams.train_name = "Supervised"
	writer = build_writer(hparams, suffix=suffix)

	if hparams.mode == "onehot":
		criterion = cross_entropy
	elif hparams.mode == "multihot":
		criterion = binary_cross_entropy
	else:
		raise RuntimeError("Invalid argument \"mode = %s\". Use %s." % (hparams.mode, " or ".join(("onehot", "multihot"))))

	trainer = SupervisedTrainer(
		model, acti_fn, optim, loader_train_full, criterion, metric_s, writer
	)
	validator = DefaultValidator(
		model, acti_fn, loader_val, metrics_val, writer
	)
	learner = DefaultLearner(hparams.train_name, trainer, validator, hparams.nb_epochs)
	learner.start()

	writer.add_hparams(hparam_dict=dict(hparams), metric_dict={})
	writer.close()


def default_supervised_hparams() -> edict:
	hparams = edict()
	hparams.lr = 1e-2
	hparams.weight_decay = 1e-4
	return hparams
