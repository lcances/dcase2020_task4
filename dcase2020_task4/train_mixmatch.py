
from easydict import EasyDict as edict
from torch.nn import Module
from torch.optim import SGD
from torch.utils.data import DataLoader
from typing import Callable, Dict

from dcase2020.pytorch_metrics.metrics import Metrics

from dcase2020_task4.mixmatch.loss import MixMatchLoss
from dcase2020_task4.mixmatch.mixer import MixMatchMixer
from dcase2020_task4.mixmatch.rampup import RampUp
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
	metric_s: Metrics,
	metric_u: Metrics,
	metrics_val: Dict[str, Metrics],
	hparams: edict,
):
	if loader_train_s.batch_size != loader_train_u.batch_size:
		raise RuntimeError("Supervised and unsupervised batch size must be equal. (%d != %d)" % (
			loader_train_s.batch_size, loader_train_u.batch_size))

	optim = SGD(model.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay)

	hparams.train_name = "MixMatch"
	writer = build_writer(hparams, suffix=hparams.criterion_unsupervised)

	nb_rampup_steps = hparams.nb_epochs * len(loader_train_u)

	criterion = MixMatchLoss.from_edict(hparams)
	mixer = MixMatchMixer(model, acti_fn, augm_fn, hparams.nb_augms, hparams.sharpen_temp, hparams.mixup_alpha)
	lambda_u_rampup = RampUp(hparams.lambda_u_max, nb_rampup_steps)

	trainer = MixMatchTrainer(
		model, acti_fn, optim, loader_train_s, loader_train_u, augm_fn, metric_s, metric_u,
		writer, criterion, mixer, lambda_u_rampup
	)
	validator = DefaultValidator(
		model, acti_fn, loader_val, metrics_val, writer
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
