
from easydict import EasyDict as edict
from time import time
from torch.nn import Module, CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from typing import Callable, List

from dcase2020.pytorch_metrics.metrics import Metrics

from dcase2020_task4.fixmatch.cosine_scheduler import CosineLRScheduler
from dcase2020_task4.fixmatch.trainer import FixMatchTrainer
from dcase2020_task4.util.utils_match import build_writer
from dcase2020_task4.learner import DefaultLearner
from dcase2020_task4.validate import DefaultValidator


def train_fixmatch(
	model: Module,
	acti_fn: Callable,
	loader_train_s: DataLoader,
	loader_train_u: DataLoader,
	loader_val: DataLoader,
	weak_augm_fn: Callable,
	strong_augm_fn: Callable,
	metrics_s: Metrics,
	metrics_u: Metrics,
	metrics_val_lst: List[Metrics],
	metrics_names: List[str],
	hparams: edict,
):
	hparams.lambda_u = 1.0
	hparams.beta = 0.9  # used only for SGD
	hparams.threshold = 0.95  # tau
	hparams.batch_size = 64  # in paper: 64
	hparams.lr0 = 0.03  # learning rate, eta
	hparams.weight_decay = 1e-4

	optim = SGD(model.parameters(), lr=hparams.lr0, weight_decay=hparams.weight_decay)
	scheduler = CosineLRScheduler(optim, nb_epochs=hparams.nb_epochs, lr0=hparams.lr0)

	hparams.train_name = "FixMatch"
	writer = build_writer(hparams)

	trainer = FixMatchTrainer(
		model, acti_fn, optim, loader_train_s, loader_train_u, weak_augm_fn, strong_augm_fn, metrics_s, metrics_u, writer, hparams
	)
	validator = DefaultValidator(
		model, acti_fn, loader_val, CrossEntropyLoss(), metrics_val_lst, metrics_names, writer, hparams.nb_classes
	)
	learner = DefaultLearner(trainer, validator, hparams.nb_epochs, scheduler)

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
