
from easydict import EasyDict as edict
from torch.nn import Module
from torch.optim import Adam
from torch.utils.data import DataLoader
from typing import Callable, Dict

from dcase2020_task4.mixmatch.loss import MixMatchLoss
from dcase2020_task4.mixmatch.mixer import MixMatchMixer
from dcase2020_task4.mixmatch.rampup import RampUp
from dcase2020_task4.mixmatch.trainer import MixMatchTrainer
from dcase2020_task4.util.utils_match import build_writer
from dcase2020_task4.learner import DefaultLearner
from dcase2020_task4.validator import DefaultValidator

from metric_utils.metrics import Metrics


def train_mixmatch(
	model: Module,
	acti_fn: Callable,
	loader_train_s_augm: DataLoader,
	loader_train_u_augms: DataLoader,
	loader_val: DataLoader,
	metrics_s: Dict[str, Metrics],
	metrics_u: Dict[str, Metrics],
	metrics_val: Dict[str, Metrics],
	hparams: edict,
):
	if loader_train_s_augm.batch_size != loader_train_u_augms.batch_size:
		raise RuntimeError("Supervised and unsupervised batch size must be equal. (%d != %d)" % (
			loader_train_s_augm.batch_size, loader_train_u_augms.batch_size))

	optim = Adam(model.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay)

	hparams.train_name = "MixMatch"
	writer = build_writer(hparams, suffix=hparams.criterion_name_u)

	nb_rampup_steps = hparams.nb_epochs * len(loader_train_u_augms)

	criterion = MixMatchLoss.from_edict(hparams)
	mixer = MixMatchMixer(model, acti_fn, hparams.nb_augms, hparams.sharpen_temp, hparams.mixup_alpha)
	lambda_u_rampup = RampUp(hparams.lambda_u_max, nb_rampup_steps)

	trainer = MixMatchTrainer(
		model, acti_fn, optim, loader_train_s_augm, loader_train_u_augms, metrics_s, metrics_u,
		writer, criterion, mixer, lambda_u_rampup
	)
	validator = DefaultValidator(
		model, acti_fn, loader_val, metrics_val, writer
	)
	learner = DefaultLearner(hparams.train_name, trainer, validator, hparams.nb_epochs)
	learner.start()

	writer.add_hparams(hparam_dict=dict(hparams), metric_dict={})
	writer.close()
