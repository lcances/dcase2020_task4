"""
	Main script for testing MixMatch, ReMixMatch, FixMatch or supervised training on a mono-label dataset.
	Available datasets are CIFAR10 and UrbanSound8k.
	They do not have a supervised/unsupervised separation, so we need to create it.
"""

import os
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

import numpy as np
import os.path as osp
import torch

from argparse import ArgumentParser, Namespace
from time import time
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import RandomChoice, Compose
from typing import Callable

from augmentation_utils.img_augmentations import Transform
from augmentation_utils.signal_augmentations import TimeStretch, PitchShiftRandom, Occlusion, Noise2
from augmentation_utils.spec_augmentations import HorizontalFlip, VerticalFlip, Noise, RandomTimeDropout, RandomFreqDropout

from dcase2020.util.utils import get_datetime, reset_seed

from dcase2020_task4.fixmatch.losses.onehot import FixMatchLossOneHot
from dcase2020_task4.fixmatch.trainer import FixMatchTrainer

from dcase2020_task4.guessers import GuesserModelOneHot, GuesserMeanModelSharpen, GuesserModelAlignmentSharpen

from dcase2020_task4.mixmatch.losses.tag.onehot import MixMatchLossOneHot
from dcase2020_task4.mixmatch.mixers.tag import MixMatchMixer
from dcase2020_task4.mixmatch.trainer import MixMatchTrainer
from dcase2020_task4.mixmatch.trainer_v3 import MixMatchTrainerV3

from dcase2020_task4.mixup.mixers.tag import MixUpMixerTag

from dcase2020_task4.remixmatch.losses.tag.onehot import ReMixMatchLossOneHot
from dcase2020_task4.remixmatch.mixers.tag import ReMixMatchMixer
from dcase2020_task4.remixmatch.self_label import SelfSupervisedFlips, SelfSupervisedRotation
from dcase2020_task4.remixmatch.trainer import ReMixMatchTrainer

from dcase2020_task4.supervised.trainer import SupervisedTrainer

from dcase2020_task4.util.avg_distributions import AvgDistributions
from dcase2020_task4.util.checkpoint import CheckPoint
from dcase2020_task4.util.dataset_idx import get_classes_idx, shuffle_classes_idx, reduce_classes_idx, split_classes_idx
from dcase2020_task4.util.FnDataset import FnDataset
from dcase2020_task4.util.MultipleDataset import MultipleDataset
from dcase2020_task4.util.NoLabelDataset import NoLabelDataset
from dcase2020_task4.util.other_augments import Gray, Inversion, RandCrop, UniColor, RandCropSpec
from dcase2020_task4.util.other_metrics import CategoricalAccuracyOnehot, MaxMetric, FnMetric, EqConfidenceMetric
from dcase2020_task4.util.ramp_up import RampUp
from dcase2020_task4.util.sharpen import Sharpen
from dcase2020_task4.util.types import str_to_bool, str_to_optional_str, str_to_union_str_int
from dcase2020_task4.util.utils_match import cross_entropy
from dcase2020_task4.util.utils_standalone import build_writer, get_nb_parameters, save_writer, model_factory, optim_factory, sched_factory, post_process_args, check_args

from dcase2020_task4.learner import DefaultLearner
from dcase2020_task4.validator import DefaultValidator

from ubs8k.datasets import Dataset as UBS8KDataset
from ubs8k.datasetManager import DatasetManager as UBS8KDatasetManager


def create_args() -> Namespace:
	parser = ArgumentParser()
	parser.add_argument("--run", type=str, default=None,
						choices=["fixmatch", "fm", "mixmatch", "mm", "remixmatch", "rmm", "supervised_full", "sf", "supervised_part", "sp"],
						help="Training method to run.")
	parser.add_argument("--seed", type=int, default=123)
	parser.add_argument("--debug_mode", type=str_to_bool, default=False)
	parser.add_argument("--suffix", type=str, default="",
						help="Suffix to Tensorboard log dir.")

	parser.add_argument("--mode", type=str, default="onehot", choices=["onehot"])
	parser.add_argument("--dataset_path", type=str, default="../dataset/CIFAR10/", required=True)
	parser.add_argument("--dataset_name", type=str, default="CIFAR10", choices=["CIFAR10", "UBS8K"])
	parser.add_argument("--nb_classes", type=int, default=10)

	parser.add_argument("--logdir", type=str, default="../../tensorboard")
	parser.add_argument("--model", type=str, default="VGG11", choices=["VGG11", "ResNet18", "UBS8KBaseline", "CNN03"])
	parser.add_argument("--nb_epochs", type=int, default=100)
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

	parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD"],
						help="Optimizer used.")
	parser.add_argument("--scheduler", type=str_to_optional_str, default="Cosine",
						choices=["CosineLRScheduler", "Cosine", "None"],
						help="FixMatch scheduler used. Use \"None\" for constant learning rate.")
	parser.add_argument("--lr", type=float, default=1e-3,
						help="Learning rate used.")
	parser.add_argument("--weight_decay", type=float, default=0.0,
						help="Weight decay used.")

	parser.add_argument("--ratio_augm_weak", type=float, default=0.5,
						help="Probability to apply weak augmentation for ReMixMatch and FixMatch.")
	parser.add_argument("--ratio_augm_strong", type=float, default=1.0,
						help="Probability to apply strong augmentation for ReMixMatch and FixMatch.")
	parser.add_argument("--ratio_augm", type=float, default=0.5,
						help="Probability to apply augmentation for MixMatch.")

	parser.add_argument("--write_results", type=str_to_bool, default=True,
						help="Write results in a tensorboard SummaryWriter.")
	parser.add_argument("--args_file", type=str_to_optional_str, default=None,
						help="Filepath to args file. Values in this JSON will overwrite other options in terminal.")
	parser.add_argument("--checkpoint_path", type=str, default="../models/",
						help="Directory path where checkpoint models will be saved.")
	parser.add_argument("--checkpoint_metric_name", type=str, default="acc",
						choices=["acc"],
						help="Metric used to compare and save best model during training.")

	parser.add_argument("--use_rampup", "--use_warmup", type=str_to_bool, default=False,
						help="Use RampUp or not for lambda_u and lambda_u1 hyperparameters.")
	parser.add_argument("--nb_rampup_epochs", type=str_to_union_str_int, default="nb_epochs",
						help="Nb of epochs when lambda_u and lambda_u1 is increase from 0 to their value."
							 "Use 0 for deactivate RampUp. Use \"nb_epochs\" for ramping up during all training.")

	parser.add_argument("--lambda_u", type=float, default=1.0,
						help="MixMatch, FixMatch and ReMixMatch \"lambda_u\" hyperparameter.")
	parser.add_argument("--lambda_u1", type=float, default=0.5,
						help="ReMixMatch \"lambda_u1\" hyperparameter.")
	parser.add_argument("--lambda_r", type=float, default=0.5,
						help="ReMixMatch \"lambda_r\" hyperparameter.")

	parser.add_argument("--nb_augms", type=int, default=2,
						help="Nb of augmentations used in MixMatch.")
	parser.add_argument("--nb_augms_strong", type=int, default=2,
						help="Nb of strong augmentations used in ReMixMatch.")
	parser.add_argument("--history_size", type=int, default=128 * 64,
						help="Nb of predictions kept in AvgDistributions used in ReMixMatch.")

	parser.add_argument("--threshold_confidence", type=float, default=0.95,
						help="FixMatch threshold for compute confidence mask in loss.")
	parser.add_argument("--criterion_name_u", type=str, default="ce", choices=["sq_diff", "cross_entropy", "ce"],
						help="MixMatch unsupervised loss component.")

	parser.add_argument("--sharpen_temperature", "--temperature", type=float, default=0.5,
						help="MixMatch and ReMixMatch hyperparameter temperature used by sharpening.")
	parser.add_argument("--mixup_alpha", "--alpha", type=float, default=0.75,
						help="MixMatch and ReMixMatch hyperparameter \"alpha\" used by MixUp.")
	parser.add_argument("--mixup_distribution_name", type=str, default="beta",
						choices=["beta", "uniform", "constant"],
						help="MixUp distribution used in MixMatch and ReMixMatch.")
	parser.add_argument("--shuffle_s_with_u", type=str_to_bool, default=True,
						help="MixMatch shuffle supervised and unsupervised data.")

	parser.add_argument("--dataset_ratio", type=float, default=1.0,
						help="Ratio of the dataset used for training.")
	parser.add_argument("--supervised_ratio", type=float, default=0.1,
						help="Supervised ratio used for split dataset.")

	parser.add_argument("--cross_validation", type=str_to_bool, default=False,
						help="Use cross validation for UBS8K dataset.")
	parser.add_argument("--fold_val", type=int, default=10,
						help="Fold used for validation in UBS8K dataset.")

	parser.add_argument("--experimental", type=str_to_optional_str, default="",
						choices=["", "None", "V3"])

	return parser.parse_args()


def main():
	start_time = time()
	start_date = get_datetime()

	args = create_args()
	args = post_process_args(args)
	check_args(args)

	print("Start match_onehot. (suffix: \"%s\")" % args.suffix)

	print(" - dataset_name: %s" % args.dataset_name)
	print(" - start_date: %s" % start_date)
	print(" - model: %s" % args.model)
	print(" - train_name: %s" % args.train_name)

	print(" - batch_size_s: %d" % args.batch_size_s)
	print(" - batch_size_u: %d" % args.batch_size_u)
	print(" - optimizer: %s" % args.optimizer)
	print(" - lr: %.2e" % args.lr)

	print(" - scheduler: %s" % args.scheduler)
	print(" - lambda_u: %.2e" % args.lambda_u)
	print(" - lambda_u1: %.2e" % args.lambda_u1)
	print(" - lambda_r: %.2e" % args.lambda_r)

	print(" - use_rampup: %s" % args.use_rampup)
	print(" - nb_rampup_epochs: %d" % args.nb_rampup_epochs)
	print(" - threshold_confidence: %.2e" % args.threshold_confidence)
	print(" - shuffle_s_with_u: %s" % args.shuffle_s_with_u)

	print(" - criterion_name_u: %s" % args.criterion_name_u)

	reset_seed(args.seed)
	torch.autograd.set_detect_anomaly(args.debug_mode)

	metrics_s = {
		"s_acc": CategoricalAccuracyOnehot(dim=1),
	}
	metrics_u = {
		"u_acc": CategoricalAccuracyOnehot(dim=1),
	}
	metrics_u1 = {
		"u1_acc": CategoricalAccuracyOnehot(dim=1),
	}
	metrics_r = {
		"r_acc": CategoricalAccuracyOnehot(dim=1),
	}
	metrics_val = {
		"acc": CategoricalAccuracyOnehot(dim=1),
		"ce": FnMetric(cross_entropy),
		"eq": EqConfidenceMetric(args.confidence, dim=1),
		"max": MaxMetric(dim=1),
	}

	acti_fn = lambda x, dim: x.softmax(dim=dim).clamp(min=2e-30)
	cross_validation_results = {}

	def run(fold_val_ubs8k: int):
		if args.dataset_name.lower() == "cifar10":
			dataset_train, dataset_val, dataset_train_augm_weak, dataset_train_augm_strong, dataset_train_augm = \
				get_cifar10_datasets(args)
		elif args.dataset_name.lower() == "ubs8k":
			dataset_train, dataset_val, dataset_train_augm_weak, dataset_train_augm_strong, dataset_train_augm = \
				get_ubs8k_datasets(args, fold_val_ubs8k)
		else:
			raise RuntimeError("Unknown dataset %s" % args.dataset_name)

		sub_loaders_ratios = [args.supervised_ratio, 1.0 - args.supervised_ratio]

		# Compute sub-indexes for split train dataset
		cls_idx_all = get_classes_idx(dataset_train, args.nb_classes)
		cls_idx_all = shuffle_classes_idx(cls_idx_all)
		cls_idx_all = reduce_classes_idx(cls_idx_all, args.dataset_ratio)
		idx_train_s, idx_train_u = split_classes_idx(cls_idx_all, sub_loaders_ratios)

		idx_val = list(range(int(len(dataset_val) * args.dataset_ratio)))

		label_one_hot = lambda item: (item[0], one_hot(torch.as_tensor(item[1]), args.nb_classes).numpy())
		dataset_train = FnDataset(dataset_train, label_one_hot)
		dataset_val = FnDataset(dataset_val, label_one_hot)

		dataset_train_augm_weak = FnDataset(dataset_train_augm_weak, label_one_hot)
		dataset_train_augm_strong = FnDataset(dataset_train_augm_strong, label_one_hot)
		dataset_train_augm = FnDataset(dataset_train_augm, label_one_hot)

		dataset_val = Subset(dataset_val, idx_val)
		loader_val = DataLoader(dataset_val, batch_size=args.batch_size_s, shuffle=False, drop_last=True)

		args_loader_train_s = dict(
			batch_size=args.batch_size_s, shuffle=True, num_workers=args.num_workers_s, drop_last=True)
		args_loader_train_u = dict(
			batch_size=args.batch_size_u, shuffle=True, num_workers=args.num_workers_u, drop_last=True)

		model = model_factory(args)
		optim = optim_factory(args, model)
		sched = sched_factory(args, optim)
		print("Model selected : %s (%d parameters)." % (args.model, get_nb_parameters(model)))

		if args.write_results:
			writer = build_writer(args, start_date, "%d" % fold_val_ubs8k)
		else:
			writer = None

		nb_rampup_steps = args.nb_rampup_epochs if args.use_rampup else 0
		rampup_lambda_u = RampUp(nb_rampup_steps, args.lambda_u, obj=None, attr_name="lambda_u")
		rampup_lambda_u1 = RampUp(nb_rampup_steps, args.lambda_u1, obj=None, attr_name="lambda_u1")
		rampup_lambda_r = RampUp(nb_rampup_steps, args.lambda_u1, obj=None, attr_name="lambda_r")

		if "fm" == args.run or "fixmatch" == args.run:
			dataset_train_s_augm_weak = Subset(dataset_train_augm_weak, idx_train_s)
			dataset_train_u_augm_weak = Subset(dataset_train_augm_weak, idx_train_u)
			dataset_train_u_augm_strong = Subset(dataset_train_augm_strong, idx_train_u)

			dataset_train_u_augm_weak = NoLabelDataset(dataset_train_u_augm_weak)
			dataset_train_u_augm_strong = NoLabelDataset(dataset_train_u_augm_strong)

			dataset_train_u_augms_weak_strong = MultipleDataset([dataset_train_u_augm_weak, dataset_train_u_augm_strong])

			loader_train_s_augm_weak = DataLoader(dataset=dataset_train_s_augm_weak, **args_loader_train_s)
			loader_train_u_augms_weak_strong = DataLoader(dataset=dataset_train_u_augms_weak_strong, **args_loader_train_u)

			criterion = FixMatchLossOneHot.from_edict(args)
			guesser = GuesserModelOneHot(model, acti_fn)

			trainer = FixMatchTrainer(
				model, acti_fn, optim, loader_train_s_augm_weak, loader_train_u_augms_weak_strong, metrics_s, metrics_u,
				criterion, writer, guesser
			)

		elif "mm" == args.run or "mixmatch" == args.run:
			dataset_train_s_augm = Subset(dataset_train_augm, idx_train_s)
			dataset_train_u_augm = Subset(dataset_train_augm, idx_train_u)

			dataset_train_u_augm = NoLabelDataset(dataset_train_u_augm)
			dataset_train_u_augms = MultipleDataset([dataset_train_u_augm] * args.nb_augms)

			if args.experimental.lower() == "v3":
				dataset_train_u = Subset(dataset_train, idx_train_u)
				dataset_train_u_augms = MultipleDataset([dataset_train_u_augms, dataset_train_u])

			loader_train_s_augm = DataLoader(dataset=dataset_train_s_augm, **args_loader_train_s)
			loader_train_u_augms = DataLoader(dataset=dataset_train_u_augms, **args_loader_train_u)

			if loader_train_s_augm.batch_size != loader_train_u_augms.batch_size:
				raise RuntimeError("Supervised and unsupervised batch size must be equal. (%d != %d)" % (
					loader_train_s_augm.batch_size, loader_train_u_augms.batch_size))

			criterion = MixMatchLossOneHot.from_edict(args)
			mixup_mixer = MixUpMixerTag.from_edict(args)
			mixer = MixMatchMixer(mixup_mixer)

			sharpen_fn = Sharpen(args.sharpen_temperature)
			guesser = GuesserMeanModelSharpen(model, acti_fn, sharpen_fn)

			if args.experimental.lower() != "v3":
				trainer = MixMatchTrainer(
					model, acti_fn, optim, loader_train_s_augm, loader_train_u_augms, metrics_s, metrics_u,
					criterion, writer, mixer, guesser
				)
			else:
				trainer = MixMatchTrainerV3(
					model, acti_fn, optim, loader_train_s_augm, loader_train_u_augms, metrics_s, metrics_u,
					criterion, writer, mixer, guesser
				)

		elif "rmm" == args.run or "remixmatch" == args.run:
			dataset_train_s_augm_strong = Subset(dataset_train_augm_strong, idx_train_s)
			dataset_train_u_augm_weak = Subset(dataset_train_augm_weak, idx_train_u)
			dataset_train_u_augm_strong = Subset(dataset_train_augm_strong, idx_train_u)

			dataset_train_u_augm_weak = NoLabelDataset(dataset_train_u_augm_weak)
			dataset_train_u_augm_strong = NoLabelDataset(dataset_train_u_augm_strong)

			dataset_train_u_strongs = MultipleDataset([dataset_train_u_augm_strong] * args.nb_augms_strong)
			dataset_train_u_weak_strongs = MultipleDataset([dataset_train_u_augm_weak, dataset_train_u_strongs])

			loader_train_s_strong = DataLoader(dataset_train_s_augm_strong, **args_loader_train_s)
			loader_train_u_augms_weak_strongs = DataLoader(dataset_train_u_weak_strongs, **args_loader_train_u)

			if loader_train_s_strong.batch_size != loader_train_u_augms_weak_strongs.batch_size:
				raise RuntimeError("Supervised and unsupervised batch size must be equal. (%d != %d)" % (
					loader_train_s_strong.batch_size, loader_train_u_augms_weak_strongs.batch_size))

			criterion = ReMixMatchLossOneHot.from_edict(args)
			rampup_lambda_u1.set_obj(criterion)
			rampup_lambda_r.set_obj(criterion)

			mixup_mixer = MixUpMixerTag.from_edict(args)
			mixer = ReMixMatchMixer(mixup_mixer, args.shuffle_s_with_u)

			sharpen_fn = Sharpen(args.sharpen_temperature)
			distributions = AvgDistributions.from_edict(args)
			guesser = GuesserModelAlignmentSharpen(model, acti_fn, distributions, sharpen_fn)

			acti_rot_fn = lambda batch, dim: batch.softmax(dim=dim).clamp(min=2e-30)

			if args.dataset_name.startswith("CIFAR10"):
				ss_transform = SelfSupervisedRotation()
			elif args.dataset_name.startswith("UBS8K"):
				ss_transform = SelfSupervisedFlips()
			else:
				raise RuntimeError("Invalid argument \"mode = %s\". Use %s." % (
					args.dataset_name, " or ".join(("CIFAR10", "UBS8K"))
				))

			trainer = ReMixMatchTrainer(
				model, acti_fn, acti_rot_fn, optim, loader_train_s_strong, loader_train_u_augms_weak_strongs,
				metrics_s, metrics_u, metrics_u1, metrics_r,
				criterion, writer, mixer, distributions, guesser, ss_transform
			)

		elif "sf" == args.run or "supervised_full" == args.run:
			dataset_train_full = Subset(dataset_train, idx_train_s + idx_train_u)
			loader_train_full = DataLoader(dataset_train_full, **args_loader_train_s)

			criterion = cross_entropy

			trainer = SupervisedTrainer(
				model, acti_fn, optim, loader_train_full, metrics_s, criterion, writer
			)

		elif "sp" == args.run or "supervised_part" == args.run:
			dataset_train_part = Subset(dataset_train, idx_train_s)
			loader_train_part = DataLoader(dataset_train_part, **args_loader_train_s)

			criterion = cross_entropy

			trainer = SupervisedTrainer(
				model, acti_fn, optim, loader_train_part, metrics_s, criterion, writer
			)

		else:
			raise RuntimeError("Unknown run %s" % args.run)

		rampup_lambda_u.set_obj(criterion)

		if args.write_results:
			filename = "%s_%s_%s.torch" % (args.model, args.train_name, args.suffix)
			filepath = osp.join(args.checkpoint_path, filename)
			checkpoint = CheckPoint(model, optim, name=filepath)
		else:
			checkpoint = None

		validator = DefaultValidator(
			model, acti_fn, loader_val, metrics_val, writer, checkpoint, args.checkpoint_metric_name
		)
		steppables = [sched, rampup_lambda_u, rampup_lambda_u1, rampup_lambda_r]
		learner = DefaultLearner(args.train_name, trainer, validator, args.nb_epochs, steppables)
		learner.start()

		if writer is not None:
			save_writer(writer, args, validator)
		validator.get_metrics_recorder().print_min_max()

		_, maxs = validator.get_metrics_recorder().get_mins_maxs()
		cross_validation_results[fold_val_ubs8k] = maxs["acc"]

	if args.cross_validation:
		for fold_val_ubs8k_ in range(1, 11):
			run(fold_val_ubs8k_)

		content = [(" %d: %f" % (fold, value)) for fold, value in cross_validation_results.items()]
		print("Cross-validation results : \n", "\n".join(content))
		print("Cross-validation mean : ", np.mean(cross_validation_results.values()))
	else:
		run(args.fold_val)

	exec_time = time() - start_time
	print("")
	print("Program started at \"%s\" and terminated at \"%s\"." % (start_date, get_datetime()))
	print("Total execution time: %.2fs" % exec_time)


def get_cifar10_augms(args: Namespace) -> (Callable, Callable, Callable):
	# Weak and strong augmentations used by FixMatch and ReMixMatch
	augm_weak_fn = RandomChoice([
		HorizontalFlip(args.ratio_augm_weak),
		VerticalFlip(args.ratio_augm_weak),
		# Transform(args.ratio_augm_weak, scale=(0.75, 1.25)),
		# Transform(args.ratio_augm_weak, rotation=(-np.pi, np.pi)),
	])
	augm_strong_fn = Compose([
		# RandomChoice([
		# Transform(args.ratio_augm_strong, scale=(0.5, 1.5)),
		# Transform(args.ratio_augm_strong, rotation=(-np.pi, np.pi)),
		# ]),
		RandomChoice([
			# Gray(args.ratio_augm_strong),
			RandCrop(args.ratio_augm_strong),
			# UniColor(args.ratio_augm_strong),
			# Inversion(args.ratio_augm_strong),
		]),
	])
	# Augmentation used by MixMatch
	augm_fn = RandomChoice([
		HorizontalFlip(args.ratio_augm),
		VerticalFlip(args.ratio_augm),
		Transform(args.ratio_augm, scale=(0.75, 1.25)),
		Transform(args.ratio_augm, rotation=(-np.pi, np.pi)),
		Gray(args.ratio_augm),
		RandCrop(args.ratio_augm, rect_max_scale=(0.2, 0.2)),
		UniColor(args.ratio_augm),
		Inversion(args.ratio_augm),
	])

	return augm_weak_fn, augm_strong_fn, augm_fn


def get_cifar10_datasets(args: Namespace) -> (Dataset, Dataset, Dataset, Dataset, Dataset):
	augm_weak_fn, augm_strong_fn, augm_fn = get_cifar10_augms(args)

	# Add preprocessing before each augmentation
	preprocess_fn = lambda img: np.array(img).transpose()  # Transpose img [3, 32, 32] to [32, 32, 3]

	# Prepare data
	dataset_train = CIFAR10(args.dataset_path, train=True, download=True, transform=preprocess_fn)
	dataset_val = CIFAR10(args.dataset_path, train=False, download=True, transform=preprocess_fn)

	dataset_train_augm_weak = CIFAR10(
		args.dataset_path, train=True, download=True, transform=Compose([preprocess_fn, augm_weak_fn]))
	dataset_train_augm_strong = CIFAR10(
		args.dataset_path, train=True, download=True, transform=Compose([preprocess_fn, augm_strong_fn]))
	dataset_train_augm = CIFAR10(
		args.dataset_path, train=True, download=True, transform=Compose([preprocess_fn, augm_fn]))

	return dataset_train, dataset_val, dataset_train_augm_weak, dataset_train_augm_strong, dataset_train_augm


def get_ubs8k_augms(args: Namespace) -> (Callable, Callable, Callable):
	# Weak and strong augmentations used by FixMatch and ReMixMatch
	augm_weak_fn = RandomChoice([
		TimeStretch(args.ratio_augm_weak),
		PitchShiftRandom(args.ratio_augm_weak, steps=(-1, 1)),
		Noise(ratio=args.ratio_augm_weak, snr=5.0),
		Noise2(args.ratio_augm_weak, noise_factor=(5.0, 5.0)),
	])
	# TODO : rem
	augm_strong_fn = Compose([
		RandomChoice([
			TimeStretch(args.ratio_augm_strong),
			PitchShiftRandom(args.ratio_augm_strong),
			Noise(ratio=args.ratio_augm_strong, snr=15.0),
			Noise2(args.ratio_augm_strong, noise_factor=(10.0, 10.0)),
		]),
		RandomChoice([
			Occlusion(args.ratio_augm_strong, max_size=1.0),
			RandomFreqDropout(args.ratio_augm_strong, dropout=0.5),
			RandomTimeDropout(args.ratio_augm_strong, dropout=0.5),
			RandCropSpec(args.ratio_augm_strong),
		]),
	])
	augm_strong_fn = RandomChoice([
		Occlusion(args.ratio_augm_strong, max_size=1.0),
		RandomFreqDropout(args.ratio_augm_strong, dropout=0.5),
		RandomTimeDropout(args.ratio_augm_strong, dropout=0.5),
		RandCropSpec(args.ratio_augm_strong),
	])

	augm_fn = RandomChoice([
		TimeStretch(args.ratio_augm),
		PitchShiftRandom(args.ratio_augm),
		Occlusion(args.ratio_augm, max_size=1.0),
		Noise(ratio=args.ratio_augm, snr=5.0),
		Noise2(args.ratio_augm, noise_factor=(5.0, 5.0)),
		RandomFreqDropout(args.ratio_augm, dropout=0.5),
		RandomTimeDropout(args.ratio_augm, dropout=0.5),
		RandCropSpec(args.ratio_augm),
	])

	return augm_weak_fn, augm_strong_fn, augm_fn


def get_ubs8k_datasets(args: Namespace, fold_val: int) -> (Dataset, Dataset, Dataset, Dataset, Dataset):
	augm_weak_fn, augm_strong_fn, augm_fn = get_ubs8k_augms(args)
	metadata_root = osp.join(args.dataset_path, "metadata")
	audio_root = osp.join(args.dataset_path, "audio")

	folds_train = list(range(1, 11))
	folds_train.remove(fold_val)
	folds_train = tuple(folds_train)
	folds_val = (fold_val,)

	manager = UBS8KDatasetManager(metadata_root, audio_root)

	# Shapes : (64, 173), (1)
	dataset_train = UBS8KDataset(manager, folds=folds_train, augments=(), cached=False)
	dataset_val = UBS8KDataset(manager, folds=folds_val, augments=(), cached=True)

	dataset_train_augm_weak = UBS8KDataset(manager, folds=folds_train, augments=(augm_weak_fn,), cached=False)
	dataset_train_augm_strong = UBS8KDataset(manager, folds=folds_train, augments=(augm_strong_fn,), cached=False)
	dataset_train_augm = UBS8KDataset(manager, folds=folds_train, augments=(augm_fn,), cached=False)

	return dataset_train, dataset_val, dataset_train_augm_weak, dataset_train_augm_strong, dataset_train_augm


if __name__ == "__main__":
	main()
