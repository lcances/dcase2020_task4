"""
	Main script for testing MixMatch, ReMixMatch, FixMatch or supervised training on a multi-label dataset.
	Only DESED dataset is available.
"""

import os
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

import json
import numpy as np
import os.path as osp
import torch

from argparse import ArgumentParser, Namespace
from time import time
from torch.nn import Module, BCELoss
from torch.optim import Adam, SGD
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision.transforms import RandomChoice, Compose
from typing import Callable

from augmentation_utils.signal_augmentations import TimeStretch, PitchShiftRandom, Occlusion, Noise2
from augmentation_utils.spec_augmentations import Noise, RandomTimeDropout, RandomFreqDropout

from dcase2020.datasetManager import DESEDManager
from dcase2020.datasets import DESEDDataset

from dcase2020_task4.dcase2019.models import dcase2019_model
from dcase2020_task4.fixmatch.losses.tag_only.v1 import FixMatchLossMultiHotV1
from dcase2020_task4.fixmatch.losses.tag_only.v2 import FixMatchLossMultiHotV2
from dcase2020_task4.fixmatch.losses.tag_only.v3 import FixMatchLossMultiHotV3
from dcase2020_task4.fixmatch.losses.tag_only.v4 import FixMatchLossMultiHotV4
from dcase2020_task4.fixmatch.trainer import FixMatchTrainer
from dcase2020_task4.fixmatch.trainer_v4 import FixMatchTrainerV4

from dcase2020_task4.other_models.weak_baseline_rot import WeakBaselineRot

from dcase2020_task4.mixmatch.losses.multihot import MixMatchLossMultiHot
from dcase2020_task4.mixmatch.mixers.tag import MixMatchMixer
from dcase2020_task4.mixmatch.trainer import MixMatchTrainer
from dcase2020_task4.mixup.mixers.tag import MixUpMixerTag
from dcase2020_task4.mixup.mixers.tag_v2 import MixUpMixerTagV2

from dcase2020_task4.remixmatch.losses.multihot import ReMixMatchLossMultiHot
from dcase2020_task4.remixmatch.mixers.tag import ReMixMatchMixer
from dcase2020_task4.remixmatch.trainer import ReMixMatchTrainer

from dcase2020_task4.supervised.trainer import SupervisedTrainer

from dcase2020_task4.util.avg_distributions import AvgDistributions
from dcase2020_task4.util.cosine_scheduler import CosineLRScheduler
from dcase2020_task4.util.FnDataset import FnDataset
from dcase2020_task4.util.MultipleDataset import MultipleDataset
from dcase2020_task4.util.NoLabelDataset import NoLabelDataset
from dcase2020_task4.util.other_metrics import BinaryConfidenceAccuracy, CategoricalAccuracyOnehot, EqConfidenceMetric, FnMetric, MaxMetric, MeanMetric
from dcase2020_task4.util.ramp_up import RampUp
from dcase2020_task4.util.sharpen import SharpenMulti
from dcase2020_task4.util.types import str_to_bool, str_to_optional_str, str_to_union_str_int
from dcase2020_task4.util.utils import reset_seed, get_datetime
from dcase2020_task4.util.utils_match import build_writer, get_nb_parameters, save_writer

from dcase2020_task4.learner import DefaultLearner
from dcase2020_task4.validator import DefaultValidator

from metric_utils.metrics import FScore


def create_args() -> Namespace:
	parser = ArgumentParser()
	parser.add_argument("--run", type=str, default="fixmatch", required=True,
						choices=["fm", "fixmatch", "mm", "mixmatch", "rmm", "remixmatch", "supervised", "su"])
	parser.add_argument("--seed", type=int, default=123)
	parser.add_argument("--debug_mode", type=str_to_bool, default=False)
	parser.add_argument("--begin_date", type=str, default=get_datetime(),
						help="Date used in SummaryWriter name.")

	parser.add_argument("--mode", type=str, default="multihot")
	parser.add_argument("--dataset", type=str, default="../dataset/DESED/")
	parser.add_argument("--dataset_name", type=str, default="DESED")
	parser.add_argument("--nb_classes", type=int, default=10)

	parser.add_argument("--logdir", type=str, default="../../tensorboard/")
	parser.add_argument("--model_name", type=str, default="WeakBaseline",
						choices=["WeakBaseline"])
	parser.add_argument("--nb_epochs", type=int, default=1)
	parser.add_argument("--confidence", type=float, default=0.5,
						help="Confidence threshold used in VALIDATION.")

	parser.add_argument("--batch_size_s", type=int, default=8,
						help="Batch size used for supervised loader.")
	parser.add_argument("--batch_size_u", type=int, default=8,
						help="Batch size used for unsupervised loader.")
	parser.add_argument("--num_workers_s", type=int, default=1,
						help="Number of workers created by supervised loader.")
	parser.add_argument("--num_workers_u", type=int, default=1,
						help="Number of workers created by unsupervised loader.")

	parser.add_argument("--optim_name", type=str, default="Adam",
						choices=["Adam", "SGD"],
						help="Optimizer used.")
	parser.add_argument("--scheduler", "--sched", type=str_to_optional_str, default="CosineLRScheduler",
						help="FixMatch scheduler used. Use \"None\" for constant learning rate.")
	parser.add_argument("--lr", type=float, default=3e-3,
						help="Learning rate used.")
	parser.add_argument("--weight_decay", type=float, default=0.0,
						help="Weight decay used.")

	parser.add_argument("--write_results", type=str_to_bool, default=True,
						help="Write results in a tensorboard SummaryWriter.")
	parser.add_argument("--suffix", type=str, default="",
						help="Suffix to Tensorboard log dir.")
	parser.add_argument("--args_file", type=str_to_optional_str, default=None,
						help="Filepath to args file. Values in this JSON will overwrite other options in terminal.")

	parser.add_argument("--use_rampup", "--use_warmup", type=str_to_bool, default=False,
						help="Use RampUp or not for lambda_u and lambda_u1 hyperparameters.")
	parser.add_argument("--nb_rampup_epochs", type=str_to_union_str_int, default="nb_epochs",
						help="Nb of epochs when lambda_u and lambda_u1 is increase from 0 to their value."
							 "Use 0 for deactivate RampUp. Use \"nb_epochs\" for ramping up during all training.")
	parser.add_argument("--use_sharpen_multihot", type=str_to_bool, default=False,
						help="Use experimental multi-hot sharpening or not for MixMatch and ReMixMatch.")

	parser.add_argument("--from_disk", type=str_to_bool, default=True,
						help="Select False if you want ot load all data into RAM.")
	parser.add_argument("--criterion_name_u", type=str, default="cross_entropy",
						choices=["sq_diff", "cross_entropy"],
						help="MixMatch unsupervised loss component.")

	parser.add_argument("--lambda_u", type=float, default=1.0,
						help="FixMatch, MixMatch and ReMixMatch \"lambda_u\" hyperparameter.")
	parser.add_argument("--lambda_u1", type=float, default=0.5,
						help="ReMixMatch \"lambda_u1\" hyperparameter.")
	parser.add_argument("--lambda_r", type=float, default=0.5,
						help="ReMixMatch \"lambda_r\" hyperparameter.")

	parser.add_argument("--nb_augms", type=int, default=2,
						help="MixMatch nb of augmentations used.")
	parser.add_argument("--nb_augms_strong", type=int, default=2,
						help="ReMixMatch nb of strong augmentations used.")
	parser.add_argument("--history_size", type=int, default=128,
						help="Nb of prediction kept in AvgDistributions used in ReMixMatch.")

	parser.add_argument("--threshold_multihot", type=float, default=0.5,
						help="FixMatch threshold used to replace argmax() in multihot mode.")
	parser.add_argument("--threshold_confidence", type=float, default=0.95,
						help="FixMatch threshold for compute confidence mask in loss.")
	parser.add_argument("--sharpen_threshold_multihot", type=float, default=0.5,
						help="MixMatch threshold for multihot sharpening.")

	parser.add_argument("--sharpen_temperature", type=float, default=0.5,
						help="MixMatch and ReMixMatch hyperparameter \"temperature\" used by sharpening.")
	parser.add_argument("--mixup_alpha", type=float, default=0.75,
						help="MixMatch and ReMixMatch hyperparameter \"alpha\" used by MixUp.")
	parser.add_argument("--mixup_distribution_name", type=str, default="beta",
						choices=["beta", "uniform", "constant"])
	parser.add_argument("--shuffle_s_with_u", type=str_to_bool, default=True,
						help="MixMatch shuffle supervised and unsupervised data.")

	parser.add_argument("--experimental", type=str_to_optional_str, default="",
						choices=["", "None", "V1", "V2", "V3", "V4"])

	return parser.parse_args()


def check_args(args: Namespace):
	if not osp.isdir(args.dataset):
		raise RuntimeError("Invalid dirpath %s" % args.dataset)

	if args.write_results:
		if not osp.isdir(args.logdir):
			raise RuntimeError("Invalid dirpath %s" % args.logdir)


def main():
	start_time = time()
	start_date = get_datetime()

	args = create_args()
	if args.args_file is not None:
		args_dict = json.load(open(args.args.file, "r"))
		args.__dict__.update(args_dict)
		args.begin_date = start_date()
	check_args(args)
	if args.nb_rampup_epochs == "nb_epochs":
		args.nb_rampup_epochs = args.nb_epochs

	print("Start match_multihot (%s)." % args.suffix)
	print("- run:", " ".join(args.run))
	print("- confidence:", args.confidence)
	print("- from_disk:", args.from_disk)
	print("- debug_mode:", args.debug_mode)
	print("- experimental:", args.experimental)

	reset_seed(args.seed)
	torch.autograd.set_detect_anomaly(args.debug_mode)

	def model_factory() -> Module:
		if args.model_name == "WeakBaseline":
			return WeakBaselineRot().cuda()
		elif args.model_name == "dcase2019":
			return dcase2019_model().cuda()
		else:
			raise RuntimeError("Invalid model %s" % args.model_name)

	def optim_factory(model_: Module) -> Optimizer:
		if args.optim_name.lower() == "adam":
			return Adam(model_.parameters(), lr=args.lr, weight_decay=args.weight_decay)
		elif args.optim_name.lower() == "sgd":
			return SGD(model_.parameters(), lr=args.lr, weight_decay=args.weight_decay)
		else:
			raise RuntimeError("Unknown optimizer %s" % str(args.optim_name))

	acti_fn = lambda batch, dim: batch.sigmoid()
	augm_weak_fn, augm_strong_fn, augm_fn = get_desed_augms()

	metrics_s = {
		"s_acc_weak": BinaryConfidenceAccuracy(args.confidence),
		"s_fscore_weak": FScore(),
	}
	metrics_u = {
		"u_acc_weak": BinaryConfidenceAccuracy(args.confidence)
	}
	metrics_u1 = {
		"u1_acc_weak": BinaryConfidenceAccuracy(args.confidence)
	}
	metrics_r = {
		"r_acc": CategoricalAccuracyOnehot(dim=1)
	}
	metrics_val = {
		"acc_weak": BinaryConfidenceAccuracy(args.confidence),
		"bce_weak": FnMetric(BCELoss(reduction="mean")),
		"eq_weak": EqConfidenceMetric(args.confidence, dim=1),
		"mean_weak": MeanMetric(dim=1),
		"max_weak": MaxMetric(dim=1),
		"fscore_weak": FScore(),
	}

	manager_s, manager_u = get_desed_managers(args)

	# Validation
	get_batch_label = lambda item: (item[0], item[1][0])
	dataset_val = DESEDDataset(manager_s, train=False, val=True, augments=[], cached=True, weak=True, strong=False)
	dataset_val = FnDataset(dataset_val, get_batch_label)
	loader_val = DataLoader(dataset_val, batch_size=args.batch_size_s, shuffle=False)

	# Datasets args
	args_dataset_train_s = dict(
		manager=manager_s, train=True, val=False, cached=True, weak=True, strong=False)
	args_dataset_train_s_augm = dict(
		manager=manager_s, train=True, val=False, cached=False, weak=True, strong=False)
	args_dataset_train_u_augm = dict(
		manager=manager_u, train=True, val=False, cached=False, weak=False, strong=False)

	# Loaders args
	args_loader_train_s = dict(
		batch_size=args.batch_size_s, shuffle=True, num_workers=args.num_workers_s, drop_last=True)
	args_loader_train_u = dict(
		batch_size=args.batch_size_u, shuffle=True, num_workers=args.num_workers_u, drop_last=True)

	suffix_tag = "TAG"

	if "fm" == args.run or "fixmatch" == args.run:
		args.train_name = "FixMatch"
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
		optim = optim_factory(model)
		print("Model selected : %s (%d parameters)." % (args.model_name, get_nb_parameters(model)))

		if args.scheduler == "CosineLRScheduler":
			scheduler = CosineLRScheduler(optim, nb_epochs=args.nb_epochs, lr0=args.lr)
		else:
			scheduler = None

		if args.experimental.lower() == "v1":
			criterion = FixMatchLossMultiHotV1.from_edict(args)
		elif args.experimental.lower() == "v2":
			criterion = FixMatchLossMultiHotV2.from_edict(args)
		elif args.experimental.lower() == "v3":
			criterion = FixMatchLossMultiHotV3.from_edict(args)
		elif args.experimental.lower() == "v4":
			criterion = FixMatchLossMultiHotV4.from_edict(args)
		else:
			raise RuntimeError("Unknown experimental mode %s" % str(args.experimental))

		if args.use_rampup:
			nb_rampup_steps = args.nb_rampup_epochs * len(loader_train_u_augms_weak_strong)
			rampup_lambda_u = RampUp(nb_rampup_steps, args.lambda_u)
		else:
			rampup_lambda_u = None

		if args.write_results:
			writer = build_writer(args, suffix="%s_%d_%d_%s_%.2f_%s" % (
				suffix_tag, args.batch_size_s, args.batch_size_u, str(args.scheduler), args.lambda_u, args.suffix))
		else:
			writer = None

		if args.experimental.lower() != "v4":
			trainer = FixMatchTrainer(
				model, acti_fn, optim, loader_train_s_augm_weak, loader_train_u_augms_weak_strong, metrics_s, metrics_u,
				criterion, writer, args.mode, rampup_lambda_u, args.threshold_multihot
			)
		else:
			trainer = FixMatchTrainerV4(
				model, acti_fn, optim, loader_train_s_augm_weak, loader_train_u_augms_weak_strong, metrics_s, metrics_u,
				criterion, writer, args.mode, args.threshold_multihot, args.nb_classes
			)

		validator = DefaultValidator(
			model, acti_fn, loader_val, metrics_val, writer
		)
		learner = DefaultLearner(args.train_name, trainer, validator, args.nb_epochs, scheduler)
		learner.start()

		if writer is not None:
			save_writer(writer, args, validator)

	elif "mm" == args.run or "mixmatch" == args.run:
		args.train_name = "MixMatch"
		dataset_train_s_augm = DESEDDataset(augments=[augm_fn], **args_dataset_train_s_augm)
		dataset_train_s_augm = FnDataset(dataset_train_s_augm, get_batch_label)

		dataset_train_u_augm = DESEDDataset(augments=[augm_fn], **args_dataset_train_u_augm)
		dataset_train_u_augm = NoLabelDataset(dataset_train_u_augm)

		dataset_train_u_augms = MultipleDataset([dataset_train_u_augm] * args.nb_augms)

		loader_train_s_augm = DataLoader(dataset=dataset_train_s_augm, **args_loader_train_s)
		loader_train_u_augms = DataLoader(dataset=dataset_train_u_augms, **args_loader_train_u)

		if loader_train_s_augm.batch_size != loader_train_u_augms.batch_size:
			raise RuntimeError("Supervised and unsupervised batch size must be equal. (%d != %d)" % (
				loader_train_s_augm.batch_size, loader_train_u_augms.batch_size))

		model = model_factory()
		optim = optim_factory(model)
		print("Model selected : %s (%d parameters)." % (args.model_name, get_nb_parameters(model)))

		criterion = MixMatchLossMultiHot.from_edict(args)
		if args.experimental.lower() == "v2":
			mixup_mixer = MixUpMixerTagV2.from_edict(args)
		else:
			mixup_mixer = MixUpMixerTag.from_edict(args)
		mixer = MixMatchMixer(mixup_mixer, args.shuffle_s_with_u)

		if args.use_sharpen_multihot:
			sharpen_fn = SharpenMulti(args.sharpen_temperature, args.sharpen_threshold_multihot)
		else:
			sharpen_fn = lambda x, dim: x

		nb_rampup_steps = args.nb_rampup_epochs * len(loader_train_u_augms)
		rampup_lambda_u = RampUp(nb_rampup_steps, args.lambda_u)

		if args.write_results:
			writer = build_writer(args, suffix="%s_%d_%d_%s_%.2f_%s" % (
				suffix_tag, args.batch_size_s, args.batch_size_u, args.criterion_name_u, args.lambda_u, args.suffix))
		else:
			writer = None

		trainer = MixMatchTrainer(
			model, acti_fn, optim, loader_train_s_augm, loader_train_u_augms, metrics_s, metrics_u,
			criterion, writer, mixer, rampup_lambda_u, sharpen_fn
		)
		validator = DefaultValidator(
			model, acti_fn, loader_val, metrics_val, writer
		)
		learner = DefaultLearner(args.train_name, trainer, validator, args.nb_epochs)
		learner.start()

		if writer is not None:
			save_writer(writer, args, validator)

	elif "rmm" == args.run or "remixmatch" == args.run:
		args.train_name = "ReMixMatch"
		dataset_train_s_augm_strong = DESEDDataset(augments=[augm_strong_fn], **args_dataset_train_s_augm)
		dataset_train_s_augm_strong = FnDataset(dataset_train_s_augm_strong, get_batch_label)

		dataset_train_u_augm_weak = DESEDDataset(augments=[augm_weak_fn], **args_dataset_train_u_augm)
		dataset_train_u_augm_weak = NoLabelDataset(dataset_train_u_augm_weak)

		dataset_train_u_augm_strong = DESEDDataset(augments=[augm_strong_fn], **args_dataset_train_u_augm)
		dataset_train_u_augm_strong = NoLabelDataset(dataset_train_u_augm_strong)

		dataset_train_u_augms_strongs = MultipleDataset([dataset_train_u_augm_strong] * args.nb_augms_strong)
		dataset_train_u_augms_weak_strongs = MultipleDataset([dataset_train_u_augm_weak, dataset_train_u_augms_strongs])

		loader_train_s_augm_strong = DataLoader(dataset=dataset_train_s_augm_strong, **args_loader_train_s)
		loader_train_u_augms_weak_strongs = DataLoader(dataset=dataset_train_u_augms_weak_strongs, **args_loader_train_u)

		if loader_train_s_augm_strong.batch_size != loader_train_u_augms_weak_strongs.batch_size:
			raise RuntimeError("Supervised and unsupervised batch size must be equal. (%d != %d)" % (
				loader_train_s_augm_strong.batch_size, loader_train_u_augms_weak_strongs.batch_size))

		rot_angles = np.array([0.0, np.pi / 2.0, np.pi, -np.pi / 2.0])

		model = model_factory()
		optim = optim_factory(model)
		print("Model selected : %s (%d parameters)." % (args.model_name, get_nb_parameters(model)))

		criterion = ReMixMatchLossMultiHot.from_edict(args)
		mixup_mixer = MixUpMixerTag.from_edict(args)
		mixer = ReMixMatchMixer(mixup_mixer)
		if args.use_sharpen_multihot:
			sharpen_fn = SharpenMulti(args.sharpen_temperature, args.sharpen_threshold_multihot)
		else:
			sharpen_fn = lambda x, dim: x

		distributions = AvgDistributions.from_edict(args)
		acti_rot_fn = lambda batch, dim: batch.softmax(dim=dim).clamp(min=2e-30)
		if args.use_rampup:
			nb_rampup_steps = args.nb_rampup_epochs * len(loader_train_u_augms_weak_strongs)
			rampup_lambda_u = RampUp(nb_rampup_steps, args.lambda_u)
			rampup_lambda_u1 = RampUp(nb_rampup_steps, args.lambda_u1)
		else:
			rampup_lambda_u = None
			rampup_lambda_u1 = None

		if args.write_results:
			writer = build_writer(args, suffix="%s_%d_%d_%.2f_%.2f_%.2f_%s" % (
				suffix_tag, args.batch_size_s, args.batch_size_u, args.lambda_u, args.lambda_u1, args.lambda_r, args.suffix))
		else:
			writer = None

		trainer = ReMixMatchTrainer(
			model, acti_fn, acti_rot_fn, optim, loader_train_s_augm_strong, loader_train_u_augms_weak_strongs,
			metrics_s, metrics_u, metrics_u1, metrics_r,
			criterion, writer, mixer, distributions, rot_angles, sharpen_fn, rampup_lambda_u, rampup_lambda_u1
		)
		validator = DefaultValidator(
			model, acti_fn, loader_val, metrics_val, writer
		)
		learner = DefaultLearner(args.train_name, trainer, validator, args.nb_epochs)
		learner.start()

		if writer is not None:
			save_writer(writer, args, validator)

	elif "su" == args.run or "supervised" == args.run:
		args.train_name = "Supervised"
		dataset_train_s = DESEDDataset(**args_dataset_train_s)
		dataset_train_s = FnDataset(dataset_train_s, get_batch_label)

		loader_train_s = DataLoader(dataset=dataset_train_s, **args_loader_train_s)

		model = model_factory()
		optim = optim_factory(model)
		print("Model selected : %s (%d parameters)." % (args.model_name, get_nb_parameters(model)))

		criterion = BCELoss(reduction="mean")

		if args.write_results:
			writer = build_writer(args, suffix="%s_%d_%d_%s" % (
				suffix_tag, args.batch_size_s, args.batch_size_u, args.suffix))
		else:
			writer = None

		trainer = SupervisedTrainer(
			model, acti_fn, optim, loader_train_s, metrics_s, criterion, writer
		)
		validator = DefaultValidator(
			model, acti_fn, loader_val, metrics_val, writer
		)
		learner = DefaultLearner(args.train_name, trainer, validator, args.nb_epochs)
		learner.start()

		if writer is not None:
			save_writer(writer, args, validator)

	else:
		raise RuntimeError("Unknown run %s" % args.run)

	validator.get_metrics_recorder().print_min_max()

	exec_time = time() - start_time
	print("")
	print("Program started at \"%s\" and terminated at \"%s\"." % (start_date, get_datetime()))
	print("Total execution time: %.2fs" % exec_time)


def get_desed_managers(args) -> (DESEDManager, DESEDManager):
	desed_metadata_root = osp.join(args.dataset, "dataset", "metadata")
	desed_audio_root = osp.join(args.dataset, "dataset", "audio")

	manager_s = DESEDManager(
		desed_metadata_root, desed_audio_root,
		from_disk=args.from_disk,
		sampling_rate=22050,
		verbose=1
	)
	manager_s.add_subset("weak")
	manager_s.add_subset("synthetic20")
	manager_s.add_subset("validation")

	manager_u = DESEDManager(
		desed_metadata_root, desed_audio_root,
		from_disk=args.from_disk,
		sampling_rate=22050,
		verbose=1
	)
	manager_u.add_subset("unlabel_in_domain")

	return manager_s, manager_u


def get_desed_augms() -> (Callable, Callable, Callable):
	# Weak and strong augmentations used by FixMatch and ReMixMatch
	ratio = 0.5
	augm_weak_fn = RandomChoice([
		TimeStretch(ratio),
		PitchShiftRandom(ratio, steps=(-1, 1)),
		Noise(ratio=ratio, snr=5.0),
		Noise2(ratio, noise_factor=(5.0, 5.0)),
	])
	ratio = 1.0
	augm_strong_fn = Compose([
		RandomChoice([
			TimeStretch(ratio),
			PitchShiftRandom(ratio),
			Noise(ratio=ratio, snr=15.0),
			Noise2(ratio, noise_factor=(10.0, 10.0)),
		]),
		RandomChoice([
			Occlusion(ratio, max_size=1.0),
			RandomFreqDropout(ratio, dropout=0.5),
			RandomTimeDropout(ratio, dropout=0.5),
		]),
	])
	ratio = 0.25
	augm_fn = RandomChoice([
		TimeStretch(ratio),
		PitchShiftRandom(ratio),
		Occlusion(ratio, max_size=1.0),
		Noise(ratio=ratio, snr=5.0),
		Noise2(ratio, noise_factor=(5.0, 5.0)),
		RandomFreqDropout(ratio, dropout=0.5),
		RandomTimeDropout(ratio, dropout=0.5),
	])

	return augm_weak_fn, augm_strong_fn, augm_fn


if __name__ == "__main__":
	main()
