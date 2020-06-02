
from easydict import EasyDict as edict
from time import time
from torch import Tensor
from torch.nn import Module
from torch.optim import SGD
from torch.utils.data import DataLoader
from typing import Callable, List

from dcase2020.pytorch_metrics.metrics import Metrics

from dcase2020_task4.mixmatch.trainer import MixMatchTrainer
from dcase2020_task4.util.utils_match import build_writer
from dcase2020_task4.learner import DefaultLearner
from dcase2020_task4.validator import DefaultValidator


def train_mixmatch(
	model: Module,
	acti_fn: Callable,
	loader_train_s: DataLoader,
	loader_train_u: DataLoader,
	loader_val: DataLoader,
	augm_fn: Callable,
	metrics_s: Metrics,
	metrics_u: Metrics,
	metrics_val_lst: List[Metrics],
	metrics_val_names: List[str],
	pre_batch_fn: Callable[[Tensor], Tensor],
	pre_labels_fn: Callable[[Tensor], Tensor],
	hparams: edict,
):
	if loader_train_s.batch_size != loader_train_u.batch_size:
		raise RuntimeError("Supervised and unsupervised batch size must be equal. (%d != %d)" % (
			loader_train_s.batch_size, loader_train_u.batch_size))

	optim = SGD(model.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay)

	hparams.train_name = "MixMatch"
	writer = build_writer(hparams, suffix=hparams.criterion_unsupervised)

	trainer = MixMatchTrainer(
		model, acti_fn, optim, loader_train_s, loader_train_u, augm_fn, metrics_s, metrics_u,
		writer, pre_batch_fn, pre_labels_fn, hparams
	)
	validator = DefaultValidator(
		model, acti_fn, loader_val, metrics_val_lst, metrics_val_names, writer, pre_batch_fn, pre_labels_fn
	)
	learner = DefaultLearner(hparams.train_name, trainer, validator, hparams.nb_epochs)
	learner.start()

	writer.add_hparams(hparam_dict=dict(hparams), metric_dict={})
	writer.close()


def default_mixmatch_hparams() -> edict:
	hparams = edict()
	hparams.nb_augms = 2
	hparams.sharpen_temp = 0.5
	hparams.mixup_alpha = 0.75
	hparams.lambda_u_max = 10.0  # In paper : 75
	hparams.lr = 1e-2
	hparams.weight_decay = 8e-4
	hparams.criterion_unsupervised = "sqdiff"  # In paper : sqdiff
	return hparams
