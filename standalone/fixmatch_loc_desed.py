import os
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

import os.path as osp
import torch

from argparse import ArgumentParser, Namespace
from easydict import EasyDict as edict
from time import time
from torch.nn import Module
from torch.nn.functional import binary_cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import RandomChoice, Compose

from augmentation_utils.img_augmentations import Transform
from augmentation_utils.signal_augmentations import TimeStretch, PitchShiftRandom, Occlusion
from augmentation_utils.spec_augmentations import Noise, RandomTimeDropout, RandomFreqDropout

from dcase2020.datasetManager import DESEDManager
from dcase2020.datasets import DESEDDataset

from dcase2020_task4.dcase2019.models import dcase2019_model

from dcase2020_task4.fixmatch.hparams import default_fixmatch_hparams
from dcase2020_task4.fixmatch.losses.multihot_loc import FixMatchLossMultiHotLoc
from dcase2020_task4.fixmatch.cosine_scheduler import CosineLRScheduler
from dcase2020_task4.fixmatch.trainer_loc import FixMatchTrainerLoc

from dcase2020_task4.learner import DefaultLearner
from dcase2020_task4.supervised.hparams import default_supervised_hparams
from dcase2020_task4.supervised.loss import weak_synth_loss
from dcase2020_task4.supervised.trainer_loc import SupervisedTrainerLoc

from dcase2020_task4.util.FnDataset import FnDataset
from dcase2020_task4.util.MultipleDataset import MultipleDataset
from dcase2020_task4.util.NoLabelDataset import NoLabelDataset
from dcase2020_task4.util.other_metrics import BinaryConfidenceAccuracy, EqConfidenceMetric, FnMetric, MaxMetric, MeanMetric
from dcase2020_task4.util.utils import reset_seed, get_datetime
from dcase2020_task4.util.utils_match import build_writer

from dcase2020_task4.validator import DefaultValidatorLoc

from metric_utils.metrics import FScore


def create_args() -> Namespace:
	bool_fn = lambda x: str(x).lower() in ['true', '1', 'yes']
	optional_str = lambda x: None if str(x).lower() == "none" else str(x)

	parser = ArgumentParser()
	# TODO : help for acronyms
	parser.add_argument("--run", type=str, nargs="*", default=["fm", "sf"], choices=["fm", "sf"])
	parser.add_argument("--logdir", type=str, default="../../tensorboard/")
	parser.add_argument("--dataset", type=str, default="../dataset/DESED/")
	parser.add_argument("--mode", type=str, default="multihot")
	parser.add_argument("--seed", type=int, default=123)
	parser.add_argument("--model_name", type=str, default="dcase2019", choices=["WeakBaseline", "dcase2019"])
	parser.add_argument("--nb_epochs", type=int, default=10)
	parser.add_argument("--batch_size", type=int, default=8)
	parser.add_argument("--nb_classes", type=int, default=10)
	parser.add_argument("--confidence", type=float, default=0.5)
	parser.add_argument("--from_disk", type=bool_fn, default=True,
						help="Select False if you want ot load all data into RAM.")
	parser.add_argument("--num_workers_s", type=int, default=1)
	parser.add_argument("--num_workers_u", type=int, default=1)

	parser.add_argument("--lr", type=float, default=1e-3,
						help="Learning rate used.")
	parser.add_argument("--scheduler", "--sched", type=optional_str, default="CosineLRScheduler",
						help="FixMatch scheduler used. Use \"None\" for constant learning rate.")

	parser.add_argument("--lambda_u", type=float, default=1.0,
						help="FixMatch, MixMatch and ReMixMatch \"lambda_u\" hyperparameter.")
	parser.add_argument("--nb_augms", type=int, default=2,
						help="MixMatch nb of augmentations used.")
	parser.add_argument("--nb_augms_strong", type=int, default=2,
						help="ReMixMatch nb of strong augmentations used.")

	parser.add_argument("--threshold_multihot", type=float, default=0.5,
						help="FixMatch threshold used to replace argmax() in multihot mode.")
	parser.add_argument("--threshold_mask", type=float, default=0.9,
						help="FixMatch threshold for compute mask in loss.")

	parser.add_argument("--suffix", type=str, default="",
						help="Suffix to Tensorboard log dir.")

	parser.add_argument("--debug_mode", type=bool_fn, default=False)
	parser.add_argument("--use_label_strong", type=bool_fn, default=True)

	return parser.parse_args()


def check_args(args: Namespace):
	pass


def main():
	prog_start = time()

	args = create_args()
	check_args(args)
	print("Start fixmatch_loc_desed.")
	print("- from_disk:", args.from_disk)
	print("- debug_mode:", args.debug_mode)

	hparams = edict()
	args_filtered = {k: (" ".join(v) if isinstance(v, list) else v) for k, v in args.__dict__.items()}
	hparams.update(args_filtered)
	# Note : some hyperparameters are overwritten when calling the training function, change this in the future
	hparams.begin_date = get_datetime()
	hparams.dataset_name = "DESED"

	reset_seed(hparams.seed)
	torch.autograd.set_detect_anomaly(args.debug_mode)

	model_factory = lambda: dcase2019_model().cuda()
	acti_fn = lambda batch, dim: batch.sigmoid()

	# Weak and strong augmentations used by FixMatch and ReMixMatch
	augm_weak_fn = RandomChoice([
		Transform(0.5, scale=(0.9, 1.1)),
		PitchShiftRandom(0.5, steps=(-2, 2)),
	])
	augm_strong_fn = Compose([
		Transform(1.0, scale=(0.9, 1.1)),
		RandomChoice([
			TimeStretch(1.0),
			PitchShiftRandom(1.0),
			Occlusion(1.0, max_size=1.0),
			Noise(ratio=1.0, snr=10.0),
			RandomFreqDropout(1.0, dropout=0.5),
			RandomTimeDropout(1.0, dropout=0.5),
		]),
	])

	metrics_s_weak = {
		"s_acc_weak": BinaryConfidenceAccuracy(hparams.confidence),
		"s_fscore_weak": FScore(),
	}
	metrics_u_weak = {"acc_u_weak": BinaryConfidenceAccuracy(hparams.confidence)}
	metrics_s_strong = {
		"s_acc_strong": BinaryConfidenceAccuracy(hparams.confidence),
		"s_fscore_strong": FScore(),
	}
	metrics_u_strong = {"acc_u_strong": BinaryConfidenceAccuracy(hparams.confidence)}
	metrics_val_weak = {
		"acc_weak": BinaryConfidenceAccuracy(hparams.confidence),
		"bce_weak": FnMetric(binary_cross_entropy),
		"eq_weak": EqConfidenceMetric(hparams.confidence),
		"mean_weak": MeanMetric(),
		"max_weak": MaxMetric(),
		"fscore_weak": FScore(),
	}
	metrics_val_strong = {
		"acc_strong": BinaryConfidenceAccuracy(hparams.confidence),
		"bce_strong": FnMetric(binary_cross_entropy),
		"eq_strong": EqConfidenceMetric(hparams.confidence),
		"mean_strong": MeanMetric(),
		"max_strong": MaxMetric(),
		"fscore_strong": FScore(),
	}

	manager_s, manager_u = get_desed_managers(hparams)

	# Validation
	get_batch_label = lambda item: (item[0], item[1][0], item[1][1])
	dataset_val = DESEDDataset(manager_s, train=False, val=True, augments=[], cached=True, weak=True, strong=hparams.use_label_strong)
	dataset_val = FnDataset(dataset_val, get_batch_label)
	loader_val = DataLoader(dataset_val, batch_size=hparams.batch_size, shuffle=False)

	# Datasets args
	args_dataset_train_s = dict(
		manager=manager_s, train=True, val=False, cached=True, weak=True, strong=hparams.use_label_strong)
	args_dataset_train_s_augm = dict(
		manager=manager_s, train=True, val=False, cached=False, weak=True, strong=hparams.use_label_strong)
	args_dataset_train_u_augm = dict(
		manager=manager_u, train=True, val=False, cached=False, weak=False, strong=False)

	# Loaders args
	args_loader_train_s = dict(
		batch_size=hparams.batch_size, shuffle=True, num_workers=hparams.num_workers_s, drop_last=True)
	args_loader_train_u = dict(
		batch_size=hparams.batch_size, shuffle=True, num_workers=hparams.num_workers_u, drop_last=True)

	if "fm" in args.run:
		hparams_fm = default_fixmatch_hparams()
		hparams_fm.update(hparams)

		dataset_train_s_augm_weak = DESEDDataset(augments=[augm_weak_fn], **args_dataset_train_s_augm)
		dataset_train_s_augm_weak = FnDataset(dataset_train_s_augm_weak, get_batch_label)

		dataset_train_u_augm_weak = DESEDDataset(augments=[augm_weak_fn], **args_dataset_train_u_augm)
		dataset_train_u_augm_weak = NoLabelDataset(dataset_train_u_augm_weak)

		dataset_train_u_augm_strong = DESEDDataset(augments=[augm_strong_fn], **args_dataset_train_u_augm)
		dataset_train_u_augm_strong = NoLabelDataset(dataset_train_u_augm_strong)

		dataset_train_u_augms_weak_strong = MultipleDataset([dataset_train_u_augm_weak, dataset_train_u_augm_strong])

		loader_train_s_augm_weak = DataLoader(dataset=dataset_train_s_augm_weak, **args_loader_train_s)
		loader_train_u_augms_weak_strong = DataLoader(dataset=dataset_train_u_augms_weak_strong, **args_loader_train_u)

		model = model_factory()
		optim = Adam(model.parameters(), lr=hparams_fm.lr, weight_decay=hparams_fm.weight_decay)
		if hparams_fm.scheduler == "CosineLRScheduler":
			scheduler = CosineLRScheduler(optim, nb_epochs=hparams_fm.nb_epochs, lr0=hparams_fm.lr)
		else:
			scheduler = None

		hparams_fm.train_name = "FixMatch"
		writer = build_writer(hparams_fm, suffix="%s_%s_%s" % ("STRONG", str(hparams_fm.scheduler), hparams_fm.suffix))

		criterion = FixMatchLossMultiHotLoc.from_edict(hparams_fm)
		trainer = FixMatchTrainerLoc(
			model, acti_fn, optim, loader_train_s_augm_weak, loader_train_u_augms_weak_strong,
			metrics_s_weak, metrics_u_weak, metrics_s_strong, metrics_u_strong,
			criterion, writer, hparams_fm.threshold_multihot
		)

		validator = DefaultValidatorLoc(
			model, acti_fn, loader_val, metrics_val_weak, metrics_val_strong, writer
		)
		learner = DefaultLearner(hparams_fm.train_name, trainer, validator, hparams_fm.nb_epochs, scheduler)
		learner.start()

		hparams_dict = {k: v if v is not None else str(v) for k, v in hparams_fm.items()}
		writer.add_hparams(hparam_dict=hparams_dict, metric_dict={})
		writer.close()

	if "sf" in args.run:
		hparams_sf = default_supervised_hparams()
		hparams_sf.update(hparams)

		dataset_train_s = DESEDDataset(**args_dataset_train_s)
		dataset_train_s = FnDataset(dataset_train_s, get_batch_label)

		loader_train_s = DataLoader(dataset=dataset_train_s, **args_loader_train_s)

		model = model_factory()
		optim = Adam(model.parameters(), lr=hparams_sf.lr, weight_decay=hparams_sf.weight_decay)

		hparams_sf.train_name = "Supervised"
		writer = build_writer(hparams_sf, suffix="%s" % "STRONG")

		criterion = weak_synth_loss

		trainer = SupervisedTrainerLoc(
			model, acti_fn, optim, loader_train_s, metrics_s_weak, metrics_s_strong, criterion, writer
		)
		validator = DefaultValidatorLoc(
			model, acti_fn, loader_val, metrics_val_weak, metrics_val_strong, writer
		)
		learner = DefaultLearner(hparams_sf.train_name, trainer, validator, hparams_sf.nb_epochs)
		learner.start()

		hparams_dict = {k: v if v is not None else str(v) for k, v in hparams_sf.items()}
		writer.add_hparams(hparam_dict=hparams_dict, metric_dict={})
		writer.close()

	exec_time = time() - prog_start
	print("")
	print("Program started at \"%s\" and terminated at \"%s\"." % (hparams.begin_date, get_datetime()))
	print("Total execution time: %.2fs" % exec_time)


def get_desed_managers(hparams: edict) -> (DESEDManager, DESEDManager):
	desed_metadata_root = osp.join(hparams.dataset, "dataset", "metadata")
	desed_audio_root = osp.join(hparams.dataset, "dataset", "audio")

	manager_s = DESEDManager(
		desed_metadata_root, desed_audio_root,
		from_disk=hparams.from_disk,
		sampling_rate=22050,
		verbose=1
	)
	manager_s.add_subset("weak")
	manager_s.add_subset("synthetic20")
	manager_s.add_subset("validation")

	manager_u = DESEDManager(
		desed_metadata_root, desed_audio_root,
		from_disk=hparams.from_disk,
		sampling_rate=22050,
		verbose=1
	)
	manager_u.add_subset("unlabel_in_domain")

	return manager_s, manager_u


if __name__ == "__main__":
	main()
