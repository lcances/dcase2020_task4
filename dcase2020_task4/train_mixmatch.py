from easydict import EasyDict as edict
from time import time
from torch.nn import Module
from torch.optim import SGD
from torch.utils.data import DataLoader
from typing import Callable, List

from dcase2020.pytorch_metrics.metrics import Metrics

from dcase2020_task4.mixmatch.trainer import MixMatchTrainer
from dcase2020_task4.util.utils_match import build_writer, cross_entropy

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
	metrics_names: List[str],
	hparams: edict,
):
	if loader_train_s.batch_size != loader_train_u.batch_size:
		raise RuntimeError("Supervised and unsupervised batch size must be equal. (%d != %d)" % (
			loader_train_s.batch_size, loader_train_u.batch_size))

	# MixMatch hyperparameters
	hparams.nb_augms = 2
	hparams.sharpen_temp = 0.5
	hparams.mixup_alpha = 0.75
	hparams.lambda_u_max = 10.0  # In paper : 75
	hparams.lr = 1e-2
	hparams.weight_decay = 8e-4
	hparams.criterion_unsupervised = "l2norm"

	optim = SGD(model.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay)

	hparams.train_name = "MixMatch"
	writer = build_writer(hparams)

	trainer = MixMatchTrainer(
		model, acti_fn, optim, loader_train_s, loader_train_u, augm_fn, metrics_s, metrics_u, writer, hparams
	)
	validator = DefaultValidator(
		model, acti_fn, loader_val, metrics_val_lst, metrics_names, writer, hparams.nb_classes
	)
	learner = DefaultLearner(trainer, validator, hparams.nb_epochs)

	print("\nStart %s training (%d epochs, %d train examples supervised, %d train examples unsupervised, "
		  "%d valid examples)..." % (
			  hparams.train_name,
			  hparams.nb_epochs,
			  trainer.nb_examples_supervised(),
			  trainer.nb_examples_unsupervised(),
			  validator.nb_examples()
		  ))
	start = time()
	learner.start()
	print("End %s training. (duration = %.2f)" % (hparams.train_name, time() - start))

	writer.add_hparams(hparam_dict=dict(hparams), metric_dict={})
	writer.close()
