
from easydict import EasyDict as edict
from time import time
from torch import Tensor
from torch.nn import Module
from torch.optim import SGD
from torch.utils.data import DataLoader
from typing import Callable, List

from dcase2020.pytorch_metrics.metrics import Metrics
from dcase2020_task4.learner import DefaultLearner
from dcase2020_task4.supervised_trainer import SupervisedTrainer
from dcase2020_task4.util.utils_match import build_writer, cross_entropy_with_logits
from dcase2020_task4.validator import DefaultValidator


def train_supervised(
	model: Module,
	acti_fn: Callable,
	loader_train_full: DataLoader,
	loader_val: DataLoader,
	metrics_s: Metrics,
	metrics_val_lst: List[Metrics],
	metrics_val_names: List[str],
	hparams: edict,
	suffix: str
):
	optim = SGD(model.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay)

	hparams.train_name = "Supervised"
	writer = build_writer(hparams, suffix=suffix)

	trainer = SupervisedTrainer(
		model, acti_fn, optim, loader_train_full, cross_entropy_with_logits, metrics_s, writer
	)
	validator = DefaultValidator(
		model, acti_fn, loader_val, metrics_val_lst, metrics_val_names, writer
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
