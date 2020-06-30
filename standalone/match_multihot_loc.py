"""
	Main script for testing MixMatch, FixMatch or supervised training on a multi-label dataset with localisation labels (also called strong labels).
	Only DESED dataset is available.
"""

import os
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

import json
import os.path as osp
import torch

from argparse import ArgumentParser, Namespace
from time import time
from torch.nn import Module
from torch.nn.functional import binary_cross_entropy
from torch.optim import Adam, SGD
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision.transforms import RandomChoice, Compose

from augmentation_utils.signal_augmentations import TimeStretch, PitchShiftRandom, Occlusion, Noise2
from augmentation_utils.spec_augmentations import Noise, RandomTimeDropout, RandomFreqDropout

from dcase2020.datasetManager import DESEDManager
from dcase2020.datasets import DESEDDataset

from dcase2020_task4.dcase2019.models import dcase2019_model

from dcase2020_task4.fixmatch.losses.tag_loc.default import FixMatchLossMultiHotLoc
from dcase2020_task4.fixmatch.losses.tag_loc.v1 import FixMatchLossMultiHotLocV1
from dcase2020_task4.fixmatch.losses.tag_loc.v2 import FixMatchLossMultiHotLocV2
from dcase2020_task4.fixmatch.losses.tag_loc.v3 import FixMatchLossMultiHotLocV3
from dcase2020_task4.fixmatch.losses.tag_loc.v5 import FixMatchLossMultiHotLocV5
from dcase2020_task4.util.cosine_scheduler import CosineLRScheduler
from dcase2020_task4.fixmatch.trainer_loc import FixMatchTrainerLoc

from dcase2020_task4.mixmatch.losses.tag_loc import MixMatchLossMultiHotLoc
from dcase2020_task4.mixmatch.mixers.tag_loc import MixMatchMixerMultiHotLoc
from dcase2020_task4.mixmatch.trainer_loc import MixMatchTrainerLoc

from dcase2020_task4.learner import DefaultLearner
from dcase2020_task4.supervised.loss import SupervisedLossLoc
from dcase2020_task4.supervised.trainer_loc import SupervisedTrainerLoc

from dcase2020_task4.util.avg_distributions import AvgDistributions
from dcase2020_task4.util.checkpoint import CheckPoint
from dcase2020_task4.util.FnDataset import FnDataset
from dcase2020_task4.util.MultipleDataset import MultipleDataset
from dcase2020_task4.util.NoLabelDataset import NoLabelDataset
from dcase2020_task4.util.other_metrics import BinaryConfidenceAccuracy, EqConfidenceMetric, FnMetric, MaxMetric, MeanMetric
from dcase2020_task4.util.ramp_up import RampUp
from dcase2020_task4.util.types import str_to_bool, str_to_optional_str
from dcase2020_task4.util.utils import reset_seed, get_datetime
from dcase2020_task4.util.utils_match import build_writer, filter_hparams, get_nb_parameters

from dcase2020_task4.validator_loc import DefaultValidatorLoc
from dcase2020_task4.other_models.weak_baseline_rot import WeakStrongBaselineRot

from metric_utils.metrics import FScore


def create_args() -> Namespace:
	parser = ArgumentParser()
	parser.add_argument("--run", type=str, nargs="*", default=["fixmatch"],
						help="Options fm = FixMatch, su = Supervised")
	parser.add_argument("--seed", type=int, default=123)
	parser.add_argument("--debug_mode", "--debug", type=str_to_bool, default=False)
	parser.add_argument("--begin_date", type=str, default=get_datetime(),
						help="Date used in SummaryWriter name.")

	parser.add_argument("--mode", type=str, default="multihot")
	parser.add_argument("--dataset", type=str, default="../dataset/DESED/")
	parser.add_argument("--dataset_name", type=str, default="DESED")
	parser.add_argument("--nb_classes", type=int, default=10)

	parser.add_argument("--logdir", type=str, default="../../tensorboard/")
	parser.add_argument("--model_name", type=str, default="dcase2019", choices=["dcase2019", "WeakStrongBaseline"])
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

	parser.add_argument("--optim_name", type=str, default="Adam", choices=["Adam", "SGD"],
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

	parser.add_argument("--path_checkpoint", type=str, default="../models/")
	parser.add_argument("--from_disk", type=str_to_bool, default=True,
						help="Select False if you want ot load all data into RAM. "
							 "It will be faster but consume a lot of RAM.")
	parser.add_argument("--experimental", type=str_to_optional_str, default="",
						choices=["", "None", "V1", "V2", "V3", "V5"],
						help="Experimental FixMatch mode.")

	parser.add_argument("--use_rampup", "--use_warmup", type=str_to_bool, default=False,
						help="Use RampUp or not for lambda_u FixMatch loss hyperparameter.")
	parser.add_argument("--use_alignment", type=str_to_bool, default=False,
						help="Use distribution alignment with FixMatch predictions.")

	parser.add_argument("--checkpoint_metric_name", type=str, default="fscore_weak",
						choices=["fscore_weak", "fscore_strong", "acc_weak", "acc_strong"],
						help="Metric used to compare and save best model during training.")

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

	parser.add_argument("--sharpen_temp", type=float, default=0.5,
						help="MixMatch and ReMixMatch hyperparameter \"temperature\" used by sharpening.")
	parser.add_argument("--mixup_alpha", type=float, default=0.75,
						help="MixMatch and ReMixMatch hyperparameter \"alpha\" used by MixUp.")

	return parser.parse_args()


def check_args(args: Namespace):
	if not osp.isdir(args.dataset):
		raise RuntimeError("Invalid dirpath %s" % args.dataset)

	if args.write_results:
		if not osp.isdir(args.logdir):
			raise RuntimeError("Invalid dirpath %s" % args.logdir)
		if not osp.isdir(args.path_checkpoint):
			raise RuntimeError("Invalid dirpath %s" % args.path_checkpoint)


def main():
	start_time = time()
	start_date = get_datetime()

	args = create_args()
	check_args(args)

	reset_seed(args.seed)
	torch.autograd.set_detect_anomaly(args.debug_mode)

	print("Start match_multihot_loc (%s)." % args.suffix)
	print("- run:", " ".join(args.run))
	print("- confidence:", args.confidence)
	print("- from_disk:", args.from_disk)
	print("- debug_mode:", args.debug_mode)
	print("- experimental:", args.experimental)
	print("- use_rampup:", args.use_rampup)
	print("- use_alignment:", args.use_alignment)
	print("- threshold_multihot:", args.threshold_multihot)
	print("- threshold_confidence:", args.threshold_confidence)

	def model_factory() -> Module:
		if args.model_name == "dcase2019":
			return dcase2019_model().cuda()
		elif args.model_name == "WeakStrongBaseline":
			return WeakStrongBaselineRot().cuda()
		else:
			raise RuntimeError("Unknown model name %s" % args.model_name)

	def optim_factory(model_: Module) -> Optimizer:
		if args.optim_name.lower() == "adam":
			return Adam(model_.parameters(), lr=args.lr, weight_decay=args.weight_decay)
		elif args.optim_name.lower() == "sgd":
			return SGD(model_.parameters(), lr=args.lr, weight_decay=args.weight_decay)
		else:
			raise RuntimeError("Unknown optimizer %s" % str(args.optim_name))

	acti_fn = lambda batch, dim: batch.sigmoid()

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
		"eq_weak": EqConfidenceMetric(args.confidence),
		"mean_weak": MeanMetric(),
		"max_weak": MaxMetric(),
		"fscore_weak": FScore(),
	}
	metrics_val_strong = {
		"acc_strong": BinaryConfidenceAccuracy(args.confidence),
		"bce_strong": FnMetric(binary_cross_entropy),
		"eq_strong": EqConfidenceMetric(args.confidence),
		"mean_strong": MeanMetric(),
		"max_strong": MaxMetric(),
		"fscore_strong": FScore(),
	}

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

	suffix_loc = "LOC"

	if "fm" in args.run or "fixmatch" in args.run:
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

		if args.use_rampup:
			rampup_lambda_u = RampUp(args.lambda_u, args.nb_epochs * len(loader_train_u_augms_weak_strong))
		else:
			rampup_lambda_u = None

		if args.use_alignment:
			distributions = AvgDistributions.from_edict(args)
		else:
			distributions = None

		if args.experimental is None:
			criterion = FixMatchLossMultiHotLoc.from_edict(args)
		elif args.experimental.lower() == "v1":
			criterion = FixMatchLossMultiHotLocV1.from_edict(args)
		elif args.experimental.lower() == "v2":
			criterion = FixMatchLossMultiHotLocV2.from_edict(args)
		elif args.experimental.lower() == "v3":
			criterion = FixMatchLossMultiHotLocV3.from_edict(args)
		elif args.experimental.lower() == "v5":
			criterion = FixMatchLossMultiHotLocV5.from_edict(args)
		else:
			raise RuntimeError("Unknown experimental mode %s" % str(args.experimental))

		if args.write_results:
			writer = build_writer(args, suffix="%s_%d_%d_%s_%s_%.2f_%.2f_%s" % (
				suffix_loc, args.batch_size_s, args.batch_size_u, str(args.scheduler), args.experimental,
				args.threshold_multihot, args.threshold_confidence, args.suffix,
			))

			checkpoint = CheckPoint(
				model, optim, name=osp.join(args.path_checkpoint, "%s_%s_%s.torch" % (
					args.model_name, args.train_name, args.suffix))
			)
		else:
			writer = None
			checkpoint = None

		trainer = FixMatchTrainerLoc(
			model, acti_fn, optim, loader_train_s_augm_weak, loader_train_u_augms_weak_strong,
			metrics_s_weak, metrics_u_weak, metrics_s_strong, metrics_u_strong,
			criterion, writer, rampup_lambda_u, args.threshold_multihot, distributions
		)
		validator = DefaultValidatorLoc(
			model, acti_fn, loader_val, metrics_val_weak, metrics_val_strong, writer, checkpoint, args.checkpoint_metric_name
		)
		learner = DefaultLearner(args.train_name, trainer, validator, args.nb_epochs, scheduler)
		learner.start()

		if writer is not None:
			with open(osp.join(writer.log_dir, "args.json"), "w") as file:
				json.dump(args, file, indent="\t")
			writer.add_hparams(hparam_dict=filter_hparams(args), metric_dict={})
			writer.flush()
			writer.close()

	if "mm" in args.run or "mixmatch" in args.run:
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

		criterion = MixMatchLossMultiHotLoc.from_edict(args)
		mixer = MixMatchMixerMultiHotLoc(
			model, acti_fn,
			args.nb_augms, args.sharpen_temp, args.mixup_alpha, args.sharpen_threshold_multihot
		)
		nb_rampup_steps = args.nb_epochs * len(loader_train_u_augms)
		lambda_u_rampup = RampUp(args.lambda_u, nb_rampup_steps)

		if args.write_results:
			writer = build_writer(args, suffix="%s_%.2f_%s" % (suffix_loc, args.lambda_u, args.suffix))
			checkpoint = CheckPoint(
				model, optim, name=osp.join(args.path_checkpoint, "%s_%s_%s.torch" % (
					args.model_name, args.train_name, args.suffix))
			)
		else:
			writer = None
			checkpoint = None

		trainer = MixMatchTrainerLoc(
			model, acti_fn, optim, loader_train_s_augm, loader_train_u_augms,
			metrics_s_weak, metrics_u_weak, metrics_s_strong, metrics_u_strong,
			criterion, writer, mixer, lambda_u_rampup
		)

		validator = DefaultValidatorLoc(
			model, acti_fn, loader_val, metrics_val_weak, metrics_val_strong, writer, checkpoint, args.checkpoint_metric_name
		)
		learner = DefaultLearner(args.train_name, trainer, validator, args.nb_epochs)
		learner.start()

		if writer is not None:
			with open(osp.join(writer.log_dir, "args.json"), "w") as file:
				json.dump(args, file, indent="\t")
			writer.add_hparams(hparam_dict=filter_hparams(args), metric_dict={})
			writer.flush()
			writer.close()

	if "su" in args.run or "supervised" in args.run:
		args.train_name = "Supervised"
		dataset_train_s = DESEDDataset(**args_dataset_train_s)
		dataset_train_s = FnDataset(dataset_train_s, get_batch_label)

		loader_train_s = DataLoader(dataset=dataset_train_s, **args_loader_train_s)

		model = model_factory()
		optim = optim_factory(model)
		print("Model selected : %s (%d parameters)." % (args.model_name, get_nb_parameters(model)))

		criterion = SupervisedLossLoc()

		if args.write_results:
			writer = build_writer(args, suffix="%s_%s" % (suffix_loc, args.suffix))
			checkpoint = CheckPoint(
				model, optim, name=osp.join(args.path_checkpoint, "%s_%s_%s.torch" % (
					args.model_name, args.train_name, args.suffix))
			)
		else:
			writer = None
			checkpoint = None

		trainer = SupervisedTrainerLoc(
			model, acti_fn, optim, loader_train_s, metrics_s_weak, metrics_s_strong, criterion, writer
		)
		validator = DefaultValidatorLoc(
			model, acti_fn, loader_val, metrics_val_weak, metrics_val_strong, writer, checkpoint, args.checkpoint_metric_name
		)
		learner = DefaultLearner(args.train_name, trainer, validator, args.nb_epochs)
		learner.start()

		if writer is not None:
			with open(osp.join(writer.log_dir, "args.json"), "w") as file:
				json.dump(args, file, indent="\t")
			writer.add_hparams(hparam_dict=filter_hparams(args), metric_dict={})
			writer.flush()
			writer.close()

	exec_time = time() - start_time
	print("")
	print("Program started at \"%s\" and terminated at \"%s\"." % (start_date, get_datetime()))
	print("Total execution time: %.2fs" % exec_time)


def get_desed_managers(args: Namespace) -> (DESEDManager, DESEDManager):
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


if __name__ == "__main__":
	main()
