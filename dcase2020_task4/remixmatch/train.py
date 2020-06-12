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
	loader_train_u_augms_weak_strongs: DataLoader,
	loader_val: DataLoader,
	metrics_s: Dict[str, Metrics],
	metrics_u: Dict[str, Metrics],
	metrics_u1: Dict[str, Metrics],
	metrics_r: Dict[str, Metrics],
	metrics_val: Dict[str, Metrics],
	hparams: edict,
):
	if loader_train_s_strong.batch_size != loader_train_u_augms_weak_strongs.batch_size:
		raise RuntimeError("Supervised and unsupervised batch size must be equal. (%d != %d)" % (
			loader_train_s_strong.batch_size, loader_train_u_augms_weak_strongs.batch_size))

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
		model, acti_fn, optim, loader_train_s_strong, loader_train_u_augms_weak_strongs, metrics_s, metrics_u,
		metrics_u1, metrics_r, criterion, writer, mixer, distributions, rot_angles
	)
	validator = DefaultValidator(
		model, acti_fn, loader_val, metrics_val, writer
	)
	learner = DefaultLearner(hparams.train_name, trainer, validator, hparams.nb_epochs)
	learner.start()

	hparams_dict = {k: v if v is not None else str(v) for k, v in hparams.items()}
	writer.add_hparams(hparam_dict=hparams_dict, metric_dict={})
	writer.close()
