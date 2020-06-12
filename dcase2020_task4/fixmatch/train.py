
from easydict import EasyDict as edict
from torch.nn import Module
from torch.optim import Adam
from torch.utils.data import DataLoader
from typing import Callable, Dict

from dcase2020_task4.fixmatch.losses.multihot import FixMatchLossMultiHot
from dcase2020_task4.fixmatch.losses.multihot_loc import FixMatchLossMultiHotLoc
from dcase2020_task4.fixmatch.cosine_scheduler import CosineLRScheduler
from dcase2020_task4.fixmatch.losses.onehot import FixMatchLossOneHot
from dcase2020_task4.fixmatch.losses.v4 import FixMatchLossMultiHotV4
from dcase2020_task4.fixmatch.trainer import FixMatchTrainer
from dcase2020_task4.fixmatch.trainer_v4 import FixMatchTrainerV4
from dcase2020_task4.fixmatch.trainer_loc import FixMatchTrainerLoc
from dcase2020_task4.util.utils_match import build_writer
from dcase2020_task4.learner import DefaultLearner
from dcase2020_task4.validator import DefaultValidator

from metric_utils.metrics import Metrics


def train_fixmatch(
	model: Module,
	acti_fn: Callable,
	loader_train_s_augm_weak: DataLoader,
	loader_train_u_augms_weak_strong: DataLoader,
	loader_val: DataLoader,
	metrics_s_weak: Dict[str, Metrics],
	metrics_u_weak: Dict[str, Metrics],
	metrics_s_strong: Dict[str, Metrics],
	metrics_u_strong: Dict[str, Metrics],
	metrics_val: Dict[str, Metrics],
	hparams: edict,
):
	optim = Adam(model.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay)
	if hparams.scheduler == "CosineLRScheduler":
		scheduler = CosineLRScheduler(optim, nb_epochs=hparams.nb_epochs, lr0=hparams.lr)
	else:
		scheduler = None

	hparams.train_name = "FixMatch"
	writer = build_writer(hparams, suffix="%s_%s" % (str(hparams.scheduler), hparams.suffix))

	if hparams.use_label_strong:
		criterion = FixMatchLossMultiHotLoc.from_edict(hparams)
		trainer = FixMatchTrainerLoc(
			model, acti_fn, optim, loader_train_s_augm_weak, loader_train_u_augms_weak_strong,
			metrics_s_weak, metrics_u_weak, metrics_s_strong, metrics_u_strong,
			criterion, writer, hparams.threshold_multihot
		)
	else:
		if hparams.mode == "onehot":
			criterion = FixMatchLossOneHot.from_edict(hparams)
		elif hparams.mode == "multihot":
			criterion = FixMatchLossMultiHot.from_edict(hparams)
		else:
			raise RuntimeError("Invalid argument \"mode = %s\". Use %s." % (hparams.mode, " or ".join(("onehot", "multihot"))))

		trainer = FixMatchTrainer(
			model, acti_fn, optim, loader_train_s_augm_weak, loader_train_u_augms_weak_strong, metrics_s_weak, metrics_u_weak,
			criterion, writer, hparams.mode, hparams.threshold_multihot
		)

	validator = DefaultValidator(
		model, acti_fn, loader_val, metrics_val, writer
	)
	learner = DefaultLearner(hparams.train_name, trainer, validator, hparams.nb_epochs, scheduler)
	learner.start()

	hparams_dict = {k: v if v is not None else str(v) for k, v in hparams.items()}
	writer.add_hparams(hparam_dict=hparams_dict, metric_dict={})
	writer.close()


def train_fixmatch_v4(
	model: Module,
	acti_fn: Callable,
	loader_train_s_augm_weak: DataLoader,
	loader_train_u_augms_weak_strong: DataLoader,
	loader_val: DataLoader,
	metrics_s: Dict[str, Metrics],
	metrics_u: Dict[str, Metrics],
	metrics_val: Dict[str, Metrics],
	hparams: edict,
):
	optim = Adam(model.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay)
	if hparams.scheduler == "CosineLRScheduler":
		scheduler = CosineLRScheduler(optim, nb_epochs=hparams.nb_epochs, lr0=hparams.lr)
	else:
		scheduler = None

	hparams.train_name = "FixMatch"
	writer = build_writer(hparams, suffix="%s_%s" % (str(hparams.scheduler), hparams.suffix))

	criterion = FixMatchLossMultiHotV4.from_edict(hparams)

	trainer = FixMatchTrainerV4(
		model, acti_fn, optim, loader_train_s_augm_weak, loader_train_u_augms_weak_strong, metrics_s, metrics_u,
		criterion, writer, hparams.mode, hparams.threshold_multihot, hparams.nb_classes
	)
	validator = DefaultValidator(
		model, acti_fn, loader_val, metrics_val, writer
	)
	learner = DefaultLearner(hparams.train_name, trainer, validator, hparams.nb_epochs, scheduler)
	learner.start()

	hparams_dict = {k: v if v is not None else str(v) for k, v in hparams.items()}
	writer.add_hparams(hparam_dict=hparams_dict, metric_dict={})
	writer.close()
