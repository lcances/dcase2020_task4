import os
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

import numpy as np
import os.path as osp
import torch

from argparse import ArgumentParser, Namespace
from easydict import EasyDict as edict
from time import time
from torch.nn import Module
from torch.nn.functional import binary_cross_entropy
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision.transforms import RandomChoice

from augmentation_utils.img_augmentations import Transform
from augmentation_utils.signal_augmentations import TimeStretch, PitchShiftRandom, Occlusion
from augmentation_utils.spec_augmentations import Noise, RandomTimeDropout, RandomFreqDropout

from dcase2020.datasetManager import DESEDManager
from dcase2020.datasets import DESEDDataset

from dcase2020_task4.dcase2019.models import dcase2019_model
from dcase2020_task4.fixmatch.cosine_scheduler import CosineLRScheduler
from dcase2020_task4.fixmatch.losses.tag_only.v1 import FixMatchLossMultiHotV1
from dcase2020_task4.fixmatch.losses.tag_only.v2 import FixMatchLossMultiHotV2
from dcase2020_task4.fixmatch.losses.tag_only.v3 import FixMatchLossMultiHotV3
from dcase2020_task4.fixmatch.losses.tag_only.v4 import FixMatchLossMultiHotV4
from dcase2020_task4.fixmatch.trainer import FixMatchTrainer
from dcase2020_task4.fixmatch.trainer_v4 import FixMatchTrainerV4

from dcase2020_task4.mixmatch.losses.multihot import MixMatchLossMultiHot
from dcase2020_task4.mixmatch.mixer import MixMatchMixer
from dcase2020_task4.mixmatch.trainer import MixMatchTrainer

from dcase2020_task4.remixmatch.losses.multihot import ReMixMatchLossMultiHot
from dcase2020_task4.remixmatch.mixer import ReMixMatchMixer
from dcase2020_task4.remixmatch.model_distributions import ModelDistributions
from dcase2020_task4.remixmatch.trainer import ReMixMatchTrainer

from dcase2020_task4.supervised.trainer import SupervisedTrainer

from dcase2020_task4.util.FnDataset import FnDataset
from dcase2020_task4.util.MultipleDataset import MultipleDataset
from dcase2020_task4.util.NoLabelDataset import NoLabelDataset
from dcase2020_task4.util.other_metrics import BinaryConfidenceAccuracy, CategoricalConfidenceAccuracy, EqConfidenceMetric, FnMetric, MaxMetric, MeanMetric
from dcase2020_task4.util.rampup import RampUp
from dcase2020_task4.util.utils import reset_seed, get_datetime
from dcase2020_task4.util.utils_match import build_writer

from dcase2020_task4.weak_baseline_rot import WeakBaselineRot
from dcase2020_task4.learner import DefaultLearner
from dcase2020_task4.validator import DefaultValidator

from metric_utils.metrics import FScore


def create_args() -> Namespace:
	bool_fn = lambda x: str(x).lower() in ['true', '1', 'yes']
	optional_str = lambda x: None if str(x).lower() == "none" else str(x)

	parser = ArgumentParser()
	# TODO : help for acronyms
	parser.add_argument("--run", type=str, nargs="*", default=["fixmatch", "mixmatch", "remixmatch", "supervised"])
	parser.add_argument("--logdir", type=str, default="../../tensorboard/")
	parser.add_argument("--dataset", type=str, default="../dataset/DESED/")
	parser.add_argument("--mode", type=str, default="multihot")
	parser.add_argument("--dataset_name", type=str, default="DESED")
	parser.add_argument("--seed", type=int, default=123)
	parser.add_argument("--model_name", type=str, default="WeakBaseline", choices=["WeakBaseline"])
	parser.add_argument("--nb_epochs", type=int, default=10)
	parser.add_argument("--batch_size_s", type=int, default=8)
	parser.add_argument("--batch_size_u", type=int, default=8)
	parser.add_argument("--nb_classes", type=int, default=10)
	parser.add_argument("--confidence", type=float, default=0.5)
	parser.add_argument("--from_disk", type=bool_fn, default=True,
						help="Select False if you want ot load all data into RAM.")
	parser.add_argument("--num_workers_s", type=int, default=1)
	parser.add_argument("--num_workers_u", type=int, default=1)
	parser.add_argument("--criterion_name_u", type=str, default="crossentropy", choices=["sqdiff", "crossentropy"],
						help="MixMatch unsupervised loss component.")

	parser.add_argument("--lr", type=float, default=3e-3,
						help="Learning rate used.")
	parser.add_argument("--weight_decay", type=float, default=0.0,
						help="Weight decay used.")
	parser.add_argument("--optim_name", type=str, default="Adam", choices=["Adam"],
						help="Optimizer used.")
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
	parser.add_argument("--sharpen_threshold_multihot", type=float, default=0.5,
						help="MixMatch threshold for multihot sharpening.")

	parser.add_argument("--sharpen_temp", type=float, default=0.5,
						help="MixMatch and ReMixMatch hyperparameter \"temperature\" used by sharpening.")
	parser.add_argument("--mixup_alpha", type=float, default=0.75,
						help="MixMatch and ReMixMatch hyperparameter \"alpha\" used by MixUp.")

	parser.add_argument("--suffix", type=str, default="",
						help="Suffix to Tensorboard log dir.")

	parser.add_argument("--debug_mode", type=bool_fn, default=False)
	parser.add_argument("--experimental", type=str, default="V2", choices=["None", "V1", "V2", "V3", "V4"])

	return parser.parse_args()


def main():
	prog_start = time()
	args = create_args()

	print("Start match_desed.")
	print("- run:", " ".join(args.run))
	print("- from_disk:", args.from_disk)
	print("- debug_mode:", args.debug_mode)
	print("- experimental:", args.experimental)

	hparams = edict()
	hparams.update({
		k: (str(v) if v is None else (" ".join(v) if isinstance(v, list) else v))
		for k, v in args.__dict__.items()
	})
	# Note : some hyperparameters are overwritten when calling the training function, change this in the future
	hparams.begin_date = get_datetime()

	reset_seed(hparams.seed)
	torch.autograd.set_detect_anomaly(args.debug_mode)

	if hparams.model_name == "WeakBaseline":
		model_factory = lambda: WeakBaselineRot().cuda()
	elif hparams.model_name == "dcase2019":
		model_factory = lambda: dcase2019_model().cuda()
	else:
		raise RuntimeError("Invalid model %s" % hparams.model_name)

	acti_fn = lambda batch, dim: batch.sigmoid()

	def optim_factory(model: Module) -> Optimizer:
		if hparams.optim_name.lower() == "adam":
			return Adam(model.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay)
		else:
			raise RuntimeError("Unknown optimizer %s" % str(hparams.optim_name))

	# Weak and strong augmentations used by FixMatch and ReMixMatch
	ratio = 0.1
	augm_weak_fn = RandomChoice([
		Transform(ratio, scale=(0.9, 1.1)),
		TimeStretch(ratio),
		PitchShiftRandom(ratio),
		Occlusion(ratio, max_size=1.0),
		Noise(ratio=ratio, snr=10.0),
		RandomFreqDropout(ratio, dropout=0.5),
		RandomTimeDropout(ratio, dropout=0.5),
	])
	ratio = 1.0
	augm_strong_fn = RandomChoice([
		Transform(ratio, scale=(0.9, 1.1)),
		TimeStretch(ratio),
		PitchShiftRandom(ratio),
		Occlusion(ratio, max_size=1.0),
		Noise(ratio=ratio, snr=10.0),
		RandomFreqDropout(ratio, dropout=0.5),
		RandomTimeDropout(ratio, dropout=0.5),
	])
	# Augmentation used by MixMatch
	ratio = 0.5
	augm_fn = RandomChoice([
		Transform(ratio, scale=(0.9, 1.1)),
		TimeStretch(ratio),
		PitchShiftRandom(ratio),
		Occlusion(ratio, max_size=1.0),
		Noise(ratio=ratio, snr=10.0),
		RandomFreqDropout(ratio, dropout=0.5),
		RandomTimeDropout(ratio, dropout=0.5),
	])

	metrics_s = {
		"s_acc_weak": BinaryConfidenceAccuracy(hparams.confidence),
		"s_fscore_weak": FScore(),
	}
	metrics_u = {
		"u_acc_weak": BinaryConfidenceAccuracy(hparams.confidence)
	}
	metrics_u1 = {
		"u1_acc_weak": BinaryConfidenceAccuracy(hparams.confidence)
	}
	metrics_r = {
		"r_acc": CategoricalConfidenceAccuracy(hparams.confidence)
	}
	metrics_val = {
		"acc_weak": BinaryConfidenceAccuracy(hparams.confidence),
		"bce_weak": FnMetric(binary_cross_entropy),
		"eq_weak": EqConfidenceMetric(hparams.confidence),
		"mean_weak": MeanMetric(),
		"max_weak": MaxMetric(),
		"fscore_weak": FScore(),
	}

	manager_s, manager_u = get_desed_managers(hparams)

	# Validation
	get_batch_label = lambda item: (item[0], item[1][0])
	dataset_val = DESEDDataset(manager_s, train=False, val=True, augments=[], cached=True, weak=True, strong=False)
	dataset_val = FnDataset(dataset_val, get_batch_label)
	loader_val = DataLoader(dataset_val, batch_size=hparams.batch_size_s, shuffle=False)

	# Datasets args
	args_dataset_train_s = dict(
		manager=manager_s, train=True, val=False, cached=True, weak=True, strong=False)
	args_dataset_train_s_augm = dict(
		manager=manager_s, train=True, val=False, cached=False, weak=True, strong=False)
	args_dataset_train_u_augm = dict(
		manager=manager_u, train=True, val=False, cached=False, weak=False, strong=False)

	# Loaders args
	args_loader_train_s = dict(
		batch_size=hparams.batch_size_s, shuffle=True, num_workers=hparams.num_workers_s, drop_last=True)
	args_loader_train_u = dict(
		batch_size=hparams.batch_size_u, shuffle=True, num_workers=hparams.num_workers_u, drop_last=True)

	if "fm" in args.run or "fixmatch" in args.run:
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

		if hparams.scheduler == "CosineLRScheduler":
			scheduler = CosineLRScheduler(optim, nb_epochs=hparams.nb_epochs, lr0=hparams.lr)
		else:
			scheduler = None

		hparams.train_name = "FixMatch"
		writer = build_writer(hparams, suffix="%s_%s" % (str(hparams.scheduler), hparams.suffix))

		if hparams.experimental.lower() == "v1":
			criterion = FixMatchLossMultiHotV1.from_edict(hparams)
		elif hparams.experimental.lower() == "v2":
			criterion = FixMatchLossMultiHotV2.from_edict(hparams)
		elif hparams.experimental.lower() == "v3":
			criterion = FixMatchLossMultiHotV3.from_edict(hparams)
		elif hparams.experimental.lower() == "v4":
			criterion = FixMatchLossMultiHotV4.from_edict(hparams)
		else:
			raise RuntimeError("Unknown experimental mode %s" % str(hparams.experimental))

		if hparams.experimental.lower() != "v4":
			trainer = FixMatchTrainer(
				model, acti_fn, optim, loader_train_s_augm_weak, loader_train_u_augms_weak_strong, metrics_s, metrics_u,
				criterion, writer, hparams.mode, hparams.threshold_multihot
			)
		else:
			trainer = FixMatchTrainerV4(
				model, acti_fn, optim, loader_train_s_augm_weak, loader_train_u_augms_weak_strong, metrics_s, metrics_u,
				criterion, writer, hparams.mode, hparams.threshold_multihot, hparams.nb_classes
			)

		validator = DefaultValidator(
			model, acti_fn, loader_val, metrics_val, writer
		)
		learner = DefaultLearner(hparams.train_name, trainer, validator, hparams.nb_epochs, scheduler)
		learner.start()

		writer.add_hparams(hparam_dict=dict(hparams), metric_dict={})
		writer.close()

	if "mm" in args.run or "mixmatch" in args.run:
		dataset_train_s_augm = DESEDDataset(augments=[augm_fn], **args_dataset_train_s_augm)
		dataset_train_s_augm = FnDataset(dataset_train_s_augm, get_batch_label)

		dataset_train_u_augm = DESEDDataset(augments=[augm_fn], **args_dataset_train_u_augm)
		dataset_train_u_augm = NoLabelDataset(dataset_train_u_augm)

		dataset_train_u_augms = MultipleDataset([dataset_train_u_augm] * hparams.nb_augms)

		loader_train_s_augm = DataLoader(dataset=dataset_train_s_augm, **args_loader_train_s)
		loader_train_u_augms = DataLoader(dataset=dataset_train_u_augms, **args_loader_train_u)

		if loader_train_s_augm.batch_size != loader_train_u_augms.batch_size:
			raise RuntimeError("Supervised and unsupervised batch size must be equal. (%d != %d)" % (
				loader_train_s_augm.batch_size, loader_train_u_augms.batch_size))

		model = model_factory()
		optim = optim_factory(model)

		hparams.train_name = "MixMatch"
		writer = build_writer(hparams, suffix="%s_%s" % (hparams.criterion_name_u, hparams.suffix))

		criterion = MixMatchLossMultiHot.from_edict(hparams)
		mixer = MixMatchMixer(
			model, acti_fn,
			hparams.nb_augms, hparams.sharpen_temp, hparams.mixup_alpha, hparams.mode, hparams.sharpen_threshold_multihot
		)
		nb_rampup_steps = hparams.nb_epochs * len(loader_train_u_augms)
		lambda_u_rampup = RampUp(hparams.lambda_u, nb_rampup_steps)

		trainer = MixMatchTrainer(
			model, acti_fn, optim, loader_train_s_augm, loader_train_u_augms, metrics_s, metrics_u,
			criterion, writer, mixer, lambda_u_rampup
		)
		validator = DefaultValidator(
			model, acti_fn, loader_val, metrics_val, writer
		)
		learner = DefaultLearner(hparams.train_name, trainer, validator, hparams.nb_epochs)
		learner.start()

		writer.add_hparams(hparam_dict=dict(hparams), metric_dict={})
		writer.close()

	if "rmm" in args.run or "remixmatch" in args.run:
		dataset_train_s_augm_strong = DESEDDataset(augments=[augm_strong_fn], **args_dataset_train_s_augm)
		dataset_train_s_augm_strong = FnDataset(dataset_train_s_augm_strong, get_batch_label)

		dataset_train_u_augm_weak = DESEDDataset(augments=[augm_weak_fn], **args_dataset_train_u_augm)
		dataset_train_u_augm_weak = NoLabelDataset(dataset_train_u_augm_weak)

		dataset_train_u_augm_strong = DESEDDataset(augments=[augm_strong_fn], **args_dataset_train_u_augm)
		dataset_train_u_augm_strong = NoLabelDataset(dataset_train_u_augm_strong)

		dataset_train_u_augms_strongs = MultipleDataset([dataset_train_u_augm_strong] * hparams.nb_augms_strong)
		dataset_train_u_augms_weak_strongs = MultipleDataset([dataset_train_u_augm_weak, dataset_train_u_augms_strongs])

		loader_train_s_augm_strong = DataLoader(dataset=dataset_train_s_augm_strong, **args_loader_train_s)
		loader_train_u_augms_weak_strongs = DataLoader(dataset=dataset_train_u_augms_weak_strongs, **args_loader_train_u)

		if loader_train_s_augm_strong.batch_size != loader_train_u_augms_weak_strongs.batch_size:
			raise RuntimeError("Supervised and unsupervised batch size must be equal. (%d != %d)" % (
				loader_train_s_augm_strong.batch_size, loader_train_u_augms_weak_strongs.batch_size))

		rot_angles = np.array([0.0, np.pi / 2.0, np.pi, -np.pi / 2.0])

		model = model_factory()
		optim = optim_factory(model)

		hparams.train_name = "ReMixMatch"
		writer = build_writer(hparams, suffix="%s" % hparams.suffix)

		criterion = ReMixMatchLossMultiHot.from_edict(hparams)
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
			hparams.mode,
			hparams.sharpen_threshold_multihot,
		)
		trainer = ReMixMatchTrainer(
			model, acti_fn, optim, loader_train_s_augm_strong, loader_train_u_augms_weak_strongs, metrics_s, metrics_u,
			metrics_u1, metrics_r, criterion, writer, mixer, distributions, rot_angles
		)
		validator = DefaultValidator(
			model, acti_fn, loader_val, metrics_val, writer
		)
		learner = DefaultLearner(hparams.train_name, trainer, validator, hparams.nb_epochs)
		learner.start()

		writer.add_hparams(hparam_dict=dict(hparams), metric_dict={})
		writer.close()

	if "su" in args.run or "supervised" in args.run:
		dataset_train_s = DESEDDataset(**args_dataset_train_s)
		dataset_train_s = FnDataset(dataset_train_s, get_batch_label)

		loader_train_s = DataLoader(dataset=dataset_train_s, **args_loader_train_s)

		model = model_factory()
		optim = optim_factory(model)

		hparams.train_name = "Supervised"
		writer = build_writer(hparams, suffix="%s" % hparams.suffix)

		criterion = binary_cross_entropy

		trainer = SupervisedTrainer(
			model, acti_fn, optim, loader_train_s, metrics_s, criterion, writer
		)
		validator = DefaultValidator(
			model, acti_fn, loader_val, metrics_val, writer
		)
		learner = DefaultLearner(hparams.train_name, trainer, validator, hparams.nb_epochs)
		learner.start()

		writer.add_hparams(hparam_dict=dict(hparams), metric_dict={})
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
