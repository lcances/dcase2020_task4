import numpy as np

from easydict import EasyDict as edict
from torch.nn import Module
from torch.optim import Adam
from torch.utils.data import DataLoader
from typing import Callable, Dict

from dcase2020_task4.learner import DefaultLearner
from dcase2020_task4.remixmatch.loss import ReMixMatchLoss
from dcase2020_task4.remixmatch.mixer import ReMixMatchMixer
from dcase2020_task4.remixmatch.model_distributions import ModelDistributions
from dcase2020_task4.remixmatch.trainer import ReMixMatchTrainer
from dcase2020_task4.util.utils_match import build_writer
from dcase2020_task4.validator import DefaultValidator

from metric_utils.metrics import Metrics


def train_remixmatch(
	model: Module,
	acti_fn: Callable,
	loader_train_s_strong: DataLoader,
	loader_train_u_weak_strongs: DataLoader,
	loader_val: DataLoader,
	metric_s: Metrics,
	metric_u: Metrics,
	metric_u1: Metrics,
	metric_r: Metrics,
	metrics_val: Dict[str, Metrics],
	hparams: edict,
):
	if loader_train_s_strong.batch_size != loader_train_u_weak_strongs.batch_size:
		raise RuntimeError("Supervised and unsupervised batch size must be equal. (%d != %d)" % (
			loader_train_s_strong.batch_size, loader_train_u_weak_strongs.batch_size))

	rot_angles = np.array([0.0, np.pi / 2.0, np.pi, -np.pi / 2.0])
	optim = Adam(model.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay)

	hparams.train_name = "ReMixMatch"
	writer = build_writer(hparams)

	criterion = ReMixMatchLoss.from_edict(hparams)
	distributions = ModelDistributions(
		history_size=hparams.history_size, nb_classes=hparams.nb_classes, names=["labeled", "unlabeled"], mode=hparams.mode
	)
	mixer = ReMixMatchMixer(
		model,
		acti_fn,
		distributions,
		hparams.nb_augms_strong,
		hparams.sharpen_temp,
		hparams.mixup_alpha,
		hparams.mode
	)
	trainer = ReMixMatchTrainer(
		model, acti_fn, optim, loader_train_s_strong, loader_train_u_weak_strongs, metric_s, metric_u,
		metric_u1, metric_r, writer, criterion, mixer, distributions, rot_angles
	)
	validator = DefaultValidator(
		model, acti_fn, loader_val, metrics_val, writer
	)
	learner = DefaultLearner(hparams.train_name, trainer, validator, hparams.nb_epochs)
	learner.start()

	writer.add_hparams(hparam_dict=dict(hparams), metric_dict={})
	writer.close()
