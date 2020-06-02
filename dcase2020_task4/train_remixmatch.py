
from easydict import EasyDict as edict
from torch.nn import Module
from torch.optim import SGD
from torch.utils.data import DataLoader
from typing import Callable, List

from dcase2020.pytorch_metrics.metrics import Metrics

from dcase2020_task4.learner import DefaultLearner
from dcase2020_task4.remixmatch.loss import ReMixMatchLoss
from dcase2020_task4.remixmatch.mixer import ReMixMatchMixer
from dcase2020_task4.remixmatch.trainer import ReMixMatchTrainer
from dcase2020_task4.util.utils_match import build_writer
from dcase2020_task4.validator import DefaultValidator


def train_remixmatch(
	model: Module,
	acti_fn: Callable,
	loader_train_s: DataLoader,
	loader_train_u: DataLoader,
	loader_val: DataLoader,
	weak_augm_fn: Callable,
	strong_augm_fn: Callable,
	metrics_s: Metrics,
	metrics_u: Metrics,
	metrics_u1: Metrics,
	metrics_r: Metrics,
	metrics_val_lst: List[Metrics],
	metrics_val_names: List[str],
	hparams: edict,
):
	if loader_train_s.batch_size != loader_train_u.batch_size:
		raise RuntimeError("Supervised and unsupervised batch size must be equal. (%d != %d)" % (
			loader_train_s.batch_size, loader_train_u.batch_size))

	optim = SGD(model.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay)

	hparams.train_name = "ReMixMatch"
	writer = build_writer(hparams)

	criterion = ReMixMatchLoss.from_edict(hparams)
	mixer = ReMixMatchMixer(
		model,
		acti_fn,
		weak_augm_fn,
		strong_augm_fn,
		hparams.nb_classes,
		hparams.nb_augms_strong,
		hparams.sharpen_temp,
		hparams.mixup_alpha,
		hparams.mode
	)
	trainer = ReMixMatchTrainer(
		model, acti_fn, optim, loader_train_s, loader_train_u, weak_augm_fn, strong_augm_fn, metrics_s, metrics_u,
		metrics_u1, metrics_r, writer, criterion, mixer
	)
	validator = DefaultValidator(
		model, acti_fn, loader_val, metrics_val_lst, metrics_val_names, writer
	)
	learner = DefaultLearner(hparams.train_name, trainer, validator, hparams.nb_epochs)
	learner.start()

	writer.add_hparams(hparam_dict=dict(hparams), metric_dict={})
	writer.close()


def default_remixmatch_hparams() -> edict:
	hparams = edict()
	hparams.nb_augms_strong = 2  # In paper : 8
	hparams.sharpen_temp = 0.5
	hparams.mixup_alpha = 0.75
	hparams.lambda_u = 1.0  # In paper : 1.5
	hparams.lambda_u1 = 0.5
	hparams.lambda_r = 0.5
	hparams.lr = 1e-2  # In paper 2e-3
	hparams.weight_decay = 1e-3  # In paper 0.02
	return hparams
