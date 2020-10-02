"""
	Main script for testing MixMatch, FixMatch or supervised training on a multi-label dataset with localisation labels (also called strong labels).
	Only DESED dataset is available.
"""

import os
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

import os.path as osp
import torch

from argparse import ArgumentParser, Namespace
from time import time
from torch.nn.functional import binary_cross_entropy
from torch.utils.data import DataLoader
from torchvision.transforms import RandomChoice, Compose
from typing import Callable

from augmentation_utils.signal_augmentations import TimeStretch, PitchShiftRandom, Occlusion, Noise2
from augmentation_utils.spec_augmentations import Noise, RandomTimeDropout, RandomFreqDropout

from dcase2020.datasetManager import DESEDManager
from dcase2020.datasets import DESEDDataset

from dcase2020_task4.fixmatch.losses.loc.v1 import FixMatchLossMultiHotLocV1
from dcase2020_task4.fixmatch.losses.loc.v2 import FixMatchLossMultiHotLocV2
from dcase2020_task4.fixmatch.losses.loc.v3 import FixMatchLossMultiHotLocV3
from dcase2020_task4.fixmatch.losses.loc.v5 import FixMatchLossMultiHotLocV5
from dcase2020_task4.fixmatch.trainer_loc import FixMatchTrainerLoc

from dcase2020_task4.mixmatch.losses.loc.v1 import MixMatchLossMultiHotLoc
from dcase2020_task4.mixmatch.mixers.loc import MixMatchMixerMultiHotLoc
from dcase2020_task4.mixmatch.trainer_loc import MixMatchTrainerLoc

from dcase2020_task4.mixup.mixers.loc import MixUpMixerLoc

from dcase2020_task4.learner import Learner
from dcase2020_task4.supervised.losses.loc import SupervisedLossLoc
from dcase2020_task4.supervised.trainer_loc import SupervisedTrainerLoc

from dcase2020_task4.util.avg_distributions import DistributionAlignment
from dcase2020_task4.util.checkpoint import CheckPoint
from dcase2020_task4.util.datasets.fn_dataset import FnDataset
from dcase2020_task4.util.datasets.multiple_dataset import MultipleDataset
from dcase2020_task4.util.datasets.no_label_dataset import NoLabelDataset
from dcase2020_task4.util.other_metrics import BinaryConfidenceAccuracy, EqConfidenceMetric, FnMetric, MaxMetric, MeanMetric
from dcase2020_task4.util.ramp_up import RampUp
from dcase2020_task4.util.sharpen import SharpenMulti
from dcase2020_task4.util.types import str_to_bool, str_to_optional_str, str_to_union_str_int
from dcase2020_task4.util.utils import reset_seed, get_datetime
from dcase2020_task4.util.utils_standalone import build_writer_from_args, get_nb_parameters, save_and_close_writer, build_model_from_args, \
	build_optim_from_args, build_sched_from_args, post_process_args, check_args, save_args

from dcase2020_task4.validator_loc import ValidatorLoc

from metric_utils.metrics import FScore


def create_args() -> Namespace:
	parser = ArgumentParser()
	parser.add_argument("--run", type=str, default="fixmatch", required=True,
						choices=["fixmatch", "fm", "mixmatch", "mm", "supervised", "su"],
						help="Training method to run.")
	parser.add_argument("--seed", type=int, default=123)
	parser.add_argument("--debug_mode", "--debug", type=str_to_bool, default=False)
	parser.add_argument("--suffix", type=str, default="",
						help="Suffix to Tensorboard log dir.")

	parser.add_argument("--mode", type=str, default="multihot")
	parser.add_argument("--dataset_path", type=str, default="../dataset/DESED/")
	parser.add_argument("--dataset_name", type=str, default="DESED_LOC")
	parser.add_argument("--nb_classes", type=int, default=10)

	parser.add_argument("--logdir", type=str, default="../../tensorboard/")
	parser.add_argument("--model", type=str, default="dcase2019", choices=["dcase2019", "WeakStrongBaseline"])
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
						choices=[None, "CosineLRScheduler", "Cosine"],
						help="FixMatch scheduler used. Use \"None\" for constant learning rate.")
	parser.add_argument("--lr", type=float, default=3e-3,
						help="Learning rate used.")
	parser.add_argument("--weight_decay", type=float, default=0.0,
						help="Weight decay used.")

	parser.add_argument("--write_results", type=str_to_bool, default=True,
						help="Write results in a tensorboard SummaryWriter.")
	parser.add_argument("--args_file", type=str_to_optional_str, default=None,
						help="Filepath to args file. Values in this JSON will overwrite other options in terminal.")
	parser.add_argument("--checkpoint_path", type=str, default="../models/",
						help="Directory path where checkpoint models will be saved.")
	parser.add_argument("--checkpoint_metric_name", type=str, default="fscore_weak",
						choices=["fscore_weak", "fscore_strong", "acc_weak", "acc_strong"],
						help="Metric used to compare and save best model during training.")

	parser.add_argument("--use_rampup", "--use_warmup", type=str_to_bool, default=False,
						help="Use RampUp or not for lambda_u and lambda_u1 hyperparameters.")
	parser.add_argument("--nb_rampup_steps", type=str_to_union_str_int, default="nb_epochs",
						help="Nb of epochs when lambda_u and lambda_u1 is increase from 0 to their value."
							 "Use 0 for deactivate RampUp. Use \"nb_epochs\" for ramping up during all training.")
	parser.add_argument("--use_alignment", type=str_to_bool, default=False,
						help="Use distribution alignment with FixMatch predictions.")

	parser.add_argument("--from_disk", type=str_to_bool, default=True,
						help="Select False if you want ot load all data into RAM. "
							 "It will be faster but consume a lot of RAM.")
	parser.add_argument("--experimental", type=str_to_optional_str, default="",
						choices=[None, "V1", "V2", "V3", "V5"],
						help="Experimental FixMatch mode.")

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
						choices=["beta", "uniform", "constant"],
						help="MixUp distribution used in MixMatch and ReMixMatch.")
	parser.add_argument("--shuffle_s_with_u", type=str_to_bool, default=True,
						help="MixMatch shuffle supervised and unsupervised data.")

	return parser.parse_args()


def main():
	start_time = time()
	start_date = get_datetime()

	args = create_args()
	args = post_process_args(args)
	check_args(args)

	reset_seed(args.seed)
	torch.autograd.set_detect_anomaly(args.debug_mode)

	print("Start match_multihot_loc. (suffix: \"%s\")" % args.suffix)

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
	print(" - nb_rampup_steps: %d" % args.nb_rampup_steps)
	print(" - threshold_confidence: %.2e" % args.threshold_confidence)
	print(" - shuffle_s_with_u: %s" % args.shuffle_s_with_u)

	acti_fn = lambda batch, dim: batch.sigmoid()
	augm_weak_fn, augm_strong_fn, augm_fn = get_desed_augms(args)

	metrics_s_weak = {
		"s_acc_weak": BinaryConfidenceAccuracy(args.confidence),
		"s_fscore_weak": FScore(),
	}
	metrics_u_weak = {
		"u_acc_weak": BinaryConfidenceAccuracy(args.confidence),
		"u_fscore_weak": FScore(),
	}
	metrics_s_strong = {
		"s_acc_strong": BinaryConfidenceAccuracy(args.confidence),
		"s_fscore_strong": FScore(),
	}
	metrics_u_strong = {
		"u_acc_strong": BinaryConfidenceAccuracy(args.confidence),
		"u_fscore_strong": FScore(),
	}

	metrics_val_weak = {
		"acc_weak": BinaryConfidenceAccuracy(args.confidence),
		"bce_weak": FnMetric(binary_cross_entropy),
		"eq_weak": EqConfidenceMetric(args.confidence, dim=1),
		"mean_weak": MeanMetric(dim=1),
		"max_weak": MaxMetric(dim=1),
		"fscore_weak": FScore(),
	}
	metrics_val_strong = {
		"acc_strong": BinaryConfidenceAccuracy(args.confidence),
		"bce_strong": FnMetric(binary_cross_entropy),
		"eq_strong": EqConfidenceMetric(args.confidence, dim=(1, 2)),
		"mean_strong": MeanMetric(dim=2),
		"max_strong": MaxMetric(dim=2),
		"fscore_strong": FScore(),
	}

	manager_s, manager_u = get_desed_managers(args)

	# Validation
	get_batch_label = lambda item: (item[0], item[1][0], item[1][1])
	dataset_val = DESEDDataset(
		manager=manager_s, train=False, val=True, cached=True, weak=True, strong=True, augments=[])
	dataset_val = FnDataset(dataset_val, get_batch_label)
	loader_val = DataLoader(dataset_val, batch_size=args.batch_size_s, shuffle=False)

	# Datasets args
	args_dataset_train_s = dict(
		manager=manager_s, train=True, val=False, cached=True, weak=True, strong=True)
	args_dataset_train_s_augm = dict(
		manager=manager_s, train=True, val=False, cached=False, weak=True, strong=True)
	args_dataset_train_u_augm = dict(
		manager=manager_u, train=True, val=False, cached=False, weak=False, strong=False)

	# Loaders args
	args_loader_train_s = dict(
		batch_size=args.batch_size_s, shuffle=True, num_workers=args.num_workers_s, drop_last=True)
	args_loader_train_u = dict(
		batch_size=args.batch_size_u, shuffle=True, num_workers=args.num_workers_u, drop_last=True)

	model = build_model_from_args(args)
	optim = build_optim_from_args(args, model)
	sched = build_sched_from_args(args, optim)
	print("Model selected : %s (%d parameters)." % (args.model, get_nb_parameters(model)))

	if args.write_results:
		writer = build_writer_from_args(args, start_date, "%s" % args.experimental)
	else:
		writer = None

	nb_rampup_steps = args.nb_rampup_steps if args.use_rampup else 0
	rampup_lambda_u = RampUp(nb_rampup_steps, args.lambda_u, obj=None, attr_name="lambda_u")

	if "fm" == args.run or "fixmatch" == args.run:
		dataset_train_s_augm_weak = DESEDDataset(augments=[augm_weak_fn], **args_dataset_train_s_augm)
		dataset_train_s_augm_weak = FnDataset(dataset_train_s_augm_weak, get_batch_label)

		dataset_train_u_augm_weak = DESEDDataset(augments=[augm_weak_fn], **args_dataset_train_u_augm)
		dataset_train_u_augm_weak = NoLabelDataset(dataset_train_u_augm_weak)

		dataset_train_u_augm_strong = DESEDDataset(augments=[augm_strong_fn], **args_dataset_train_u_augm)
		dataset_train_u_augm_strong = NoLabelDataset(dataset_train_u_augm_strong)

		dataset_train_u_augms_weak_strong = MultipleDataset([dataset_train_u_augm_weak, dataset_train_u_augm_strong])

		loader_train_s_augm_weak = DataLoader(dataset=dataset_train_s_augm_weak, **args_loader_train_s)
		loader_train_u_augms_weak_strong = DataLoader(dataset=dataset_train_u_augms_weak_strong, **args_loader_train_u)

		if args.use_alignment:
			distributions = DistributionAlignment.from_args(args)
		else:
			distributions = None

		if args.experimental is None:
			criterion = FixMatchLossMultiHotLocV1.from_args(args)
		elif args.experimental == "V1":
			criterion = FixMatchLossMultiHotLocV1.from_args(args)
		elif args.experimental == "V2":
			criterion = FixMatchLossMultiHotLocV2.from_args(args)
		elif args.experimental == "V3":
			criterion = FixMatchLossMultiHotLocV3.from_args(args)
		elif args.experimental == "V5":
			criterion = FixMatchLossMultiHotLocV5.from_args(args)
		else:
			raise RuntimeError("Unknown experimental mode \"%s\"." % args.experimental)

		trainer = FixMatchTrainerLoc(
			model, acti_fn, optim, loader_train_s_augm_weak, loader_train_u_augms_weak_strong,
			metrics_s_weak, metrics_u_weak, metrics_s_strong, metrics_u_strong,
			criterion, writer, distributions, args.threshold_multihot
		)

	elif "mm" == args.run or "mixmatch" == args.run:
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

		criterion = MixMatchLossMultiHotLoc.from_args(args)
		mixup_mixer = MixUpMixerLoc.from_args(args)
		mixer = MixMatchMixerMultiHotLoc(mixup_mixer)

		if args.use_sharpen_multihot:
			sharpen_fn = SharpenMulti(args.sharpen_temperature, args.sharpen_threshold_multihot)
		else:
			sharpen_fn = lambda x, dim: x

		trainer = MixMatchTrainerLoc(
			model, acti_fn, optim, loader_train_s_augm, loader_train_u_augms,
			metrics_s_weak, metrics_u_weak, metrics_s_strong, metrics_u_strong,
			criterion, writer, mixer, sharpen_fn
		)

	elif "su" == args.run or "supervised" == args.run:
		dataset_train_s = DESEDDataset(**args_dataset_train_s)
		dataset_train_s = FnDataset(dataset_train_s, get_batch_label)

		loader_train_s = DataLoader(dataset=dataset_train_s, **args_loader_train_s)

		criterion = SupervisedLossLoc()

		trainer = SupervisedTrainerLoc(
			model, acti_fn, optim, loader_train_s, metrics_s_weak, metrics_s_strong, criterion, writer
		)

	else:
		raise RuntimeError("Unknown run %s" % args.run)

	if rampup_lambda_u is not None:
		rampup_lambda_u.set_obj(criterion)

	if args.write_results:
		filename = "%s_%s_%s.torch" % (args.model, args.train_name, args.suffix)
		filepath = osp.join(args.checkpoint_path, filename)
		checkpoint = CheckPoint(model, optim, name=filepath)
	else:
		checkpoint = None

	validator = ValidatorLoc(
		model, acti_fn, loader_val, metrics_val_weak, metrics_val_strong, writer, checkpoint, args.checkpoint_metric_name
	)
	steppables = [rampup_lambda_u, sched]
	steppables = [steppable for steppable in steppables if steppable is not None]

	learner = Learner(args.train_name, trainer, validator, args.nb_epochs, steppables)
	learner.start()

	if writer is not None:
		save_and_close_writer(writer, args)
		filepath = osp.join(writer.log_dir, "args.json")
		save_args(filepath, args)

	validator.get_metrics_recorder().print_min_max()

	exec_time = time() - start_time
	print("")
	print("Program started at \"%s\" and terminated at \"%s\"." % (start_date, get_datetime()))
	print("Total execution time: %.2fs" % exec_time)


def get_desed_managers(args: Namespace) -> (DESEDManager, DESEDManager):
	desed_metadata_root = osp.join(args.dataset_path, "dataset", "metadata")
	desed_audio_root = osp.join(args.dataset_path, "dataset", "audio")

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


def get_desed_augms(args: Namespace) -> (Callable, Callable, Callable):
	# Weak and strong augmentations used by FixMatch and ReMixMatch
	augm_weak_fn = RandomChoice([
		TimeStretch(args.ratio_augm_weak),
		PitchShiftRandom(args.ratio_augm_weak, steps=(-1, 1)),
		Noise(ratio=args.ratio_augm_weak, snr=5.0),
		Noise2(args.ratio_augm_weak, noise_factor=(5.0, 5.0)),
	])
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
		]),
	])
	augm_fn = RandomChoice([
		TimeStretch(args.ratio_augm),
		PitchShiftRandom(args.ratio_augm),
		Occlusion(args.ratio_augm, max_size=1.0),
		Noise(ratio=args.ratio_augm, snr=5.0),
		Noise2(args.ratio_augm, noise_factor=(5.0, 5.0)),
		RandomFreqDropout(args.ratio_augm, dropout=0.5),
		RandomTimeDropout(args.ratio_augm, dropout=0.5),
	])

	return augm_weak_fn, augm_strong_fn, augm_fn


def get_desed_augms_old(args: Namespace) -> (Callable, Callable, Callable):
	# TODO : rem ?
	# Weak and strong augmentations used by FixMatch and ReMixMatch
	ratio = 0.1
	augm_weak_fn = RandomChoice([
		TimeStretch(ratio),
		PitchShiftRandom(ratio),
		Occlusion(ratio, max_size=1.0),
		Noise(ratio=ratio, snr=15.0),
		Noise2(ratio, noise_factor=(10.0, 10.0)),
		RandomFreqDropout(ratio, dropout=0.5),
		RandomTimeDropout(ratio, dropout=0.5),
	])
	ratio = 0.5
	augm_strong_fn = Compose([
		TimeStretch(ratio),
		PitchShiftRandom(ratio),
		Occlusion(ratio, max_size=1.0),
		Noise(ratio=ratio, snr=15.0),
		Noise2(ratio, noise_factor=(10.0, 10.0)),
		RandomFreqDropout(ratio, dropout=0.5),
		RandomTimeDropout(ratio, dropout=0.5),
	])
	ratio = 0.5
	augm_fn = RandomChoice([
		TimeStretch(ratio),
		PitchShiftRandom(ratio),
		Occlusion(ratio, max_size=1.0),
		Noise(ratio=ratio, snr=15.0),
		Noise2(ratio, noise_factor=(10.0, 10.0)),
		RandomFreqDropout(ratio, dropout=0.5),
		RandomTimeDropout(ratio, dropout=0.5),
	])

	return augm_weak_fn, augm_strong_fn, augm_fn


if __name__ == "__main__":
	main()
