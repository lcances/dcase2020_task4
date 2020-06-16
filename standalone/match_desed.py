import os
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

import os.path as osp
import torch

from argparse import ArgumentParser, Namespace
from easydict import EasyDict as edict
from time import time
from torch.nn.functional import binary_cross_entropy
from torch.utils.data import DataLoader
from torchvision.transforms import RandomChoice, Compose

from augmentation_utils.img_augmentations import Transform
from augmentation_utils.signal_augmentations import TimeStretch, PitchShiftRandom, Occlusion
from augmentation_utils.spec_augmentations import Noise, RandomTimeDropout, RandomFreqDropout

from dcase2020.datasetManager import DESEDManager
from dcase2020.datasets import DESEDDataset

from dcase2020_task4.dcase2019.models import dcase2019_model
from dcase2020_task4.fixmatch.hparams import default_fixmatch_hparams
from dcase2020_task4.fixmatch.train import train_fixmatch
from dcase2020_task4.mixmatch.hparams import default_mixmatch_hparams
from dcase2020_task4.mixmatch.train import train_mixmatch
from dcase2020_task4.remixmatch.hparams import default_remixmatch_hparams
from dcase2020_task4.remixmatch.train import train_remixmatch
from dcase2020_task4.supervised.hparams import default_supervised_hparams
from dcase2020_task4.supervised.train import train_supervised

from dcase2020_task4.util.FnDataset import FnDataset
from dcase2020_task4.util.MultipleDataset import MultipleDataset
from dcase2020_task4.util.NoLabelDataset import NoLabelDataset
from dcase2020_task4.util.other_metrics import BinaryConfidenceAccuracy, CategoricalConfidenceAccuracy, EqConfidenceMetric, FnMetric, MaxMetric, MeanMetric
from dcase2020_task4.util.utils import reset_seed, get_datetime
from dcase2020_task4.weak_baseline_rot import WeakBaselineRot

from metric_utils.metrics import FScore


def create_args() -> Namespace:
	bool_fn = lambda x: str(x).lower() in ['true', '1', 'yes']
	optional_str = lambda x: None if str(x).lower() == "none" else str(x)

	parser = ArgumentParser()
	# TODO : help for acronyms
	parser.add_argument("--run", type=str, nargs="*", default=["fm", "mm", "rmm", "sf"], choices=["fm", "mm", "rmm", "sf"])
	parser.add_argument("--logdir", type=str, default="../../tensorboard/")
	parser.add_argument("--dataset", type=str, default="../dataset/DESED/")
	parser.add_argument("--mode", type=str, default="multihot")
	parser.add_argument("--seed", type=int, default=123)
	parser.add_argument("--model_name", type=str, default="WeakBaseline", choices=["WeakBaseline"])
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

	return parser.parse_args()


def main():
	prog_start = time()

	args = create_args()
	check_args(args)
	print("Start match_desed.")
	print("- from_disk:", args.from_disk)

	hparams = edict()
	args_filtered = {k: (" ".join(v) if isinstance(v, list) else v) for k, v in args.__dict__.items()}
	hparams.update(args_filtered)
	# Note : some hyperparameters are overwritten when calling the training function, change this in the future
	hparams.begin_date = get_datetime()
	hparams.dataset_name = "DESED"

	reset_seed(hparams.seed)
	torch.autograd.set_detect_anomaly(args.debug_mode)

	if hparams.model_name == "WeakBaseline":
		model_factory = lambda: WeakBaselineRot().cuda()
	elif hparams.model_name == "dcase2019":
		model_factory = lambda: dcase2019_model().cuda()
	else:
		raise RuntimeError("Invalid model %s" % hparams.model_name)
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
	# Augmentation used by MixMatch
	ratio = 0.5
	augm_fn = RandomChoice([
		Transform(ratio, scale=(0.9, 1.1)),
		TimeStretch(ratio),
		PitchShiftRandom(ratio),
		Noise(ratio, snr=10.0),
		Occlusion(ratio),
	])

	metrics_s = {
		"s_acc_weak": BinaryConfidenceAccuracy(hparams.confidence),
		"s_fscore_weak": FScore(),
	}
	metrics_u = {"acc_u": BinaryConfidenceAccuracy(hparams.confidence)}
	metrics_u1 = {"acc_u1": BinaryConfidenceAccuracy(hparams.confidence)}
	metrics_r = {"acc_r": CategoricalConfidenceAccuracy(hparams.confidence)}
	metrics_val = {
		"acc": BinaryConfidenceAccuracy(hparams.confidence),
		"bce": FnMetric(binary_cross_entropy),
		"eq": EqConfidenceMetric(hparams.confidence),
		"mean": MeanMetric(),
		"max": MaxMetric(),
		"fscore": FScore(),
	}

	manager_s, manager_u = get_desed_managers(hparams)

	# Validation
	get_batch_label = lambda item: (item[0], item[1][0])
	dataset_val = DESEDDataset(manager_s, train=False, val=True, augments=[], cached=True, weak=True, strong=False)
	dataset_val = FnDataset(dataset_val, get_batch_label)
	loader_val = DataLoader(dataset_val, batch_size=hparams.batch_size, shuffle=False)

	# Datasets args
	args_dataset_train_s = dict(
		manager=manager_s, train=True, val=False, cached=True, weak=True, strong=False)
	args_dataset_train_s_augm = dict(
		manager=manager_s, train=True, val=False, cached=False, weak=True, strong=False)
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

		train_fixmatch(
			model_factory(), acti_fn, loader_train_s_augm_weak, loader_train_u_augms_weak_strong, loader_val,
			metrics_s, metrics_u, metrics_val, hparams_fm
		)

	if "mm" in args.run:
		hparams_mm = default_mixmatch_hparams()
		hparams_mm.update(hparams)

		dataset_train_s_augm = DESEDDataset(augments=[augm_fn], **args_dataset_train_s_augm)
		dataset_train_s_augm = FnDataset(dataset_train_s_augm, get_batch_label)

		dataset_train_u_augm = DESEDDataset(augments=[augm_fn], **args_dataset_train_u_augm)
		dataset_train_u_augm = NoLabelDataset(dataset_train_u_augm)

		dataset_train_u_augms = MultipleDataset([dataset_train_u_augm] * hparams.nb_augms)

		loader_train_s_augm = DataLoader(dataset=dataset_train_s_augm, **args_loader_train_s)
		loader_train_u_augms = DataLoader(dataset=dataset_train_u_augms, **args_loader_train_u)

		# Train MixMatch with sqdiff for loss_u
		hparams_mm.criterion_name_u = "sqdiff"
		train_mixmatch(
			model_factory(), acti_fn, loader_train_s_augm, loader_train_u_augms, loader_val,
			metrics_s, metrics_u, metrics_val, hparams_mm
		)
		# Train MixMatch with crossentropy for loss_u
		hparams_mm.criterion_name_u = "crossentropy"
		train_mixmatch(
			model_factory(), acti_fn, loader_train_s_augm, loader_train_u_augms, loader_val,
			metrics_s, metrics_u, metrics_val, hparams_mm
		)

	if "rmm" in args.run:
		hparams_rmm = default_remixmatch_hparams()
		hparams_rmm.update(hparams)

		dataset_train_s_augm_strong = DESEDDataset(augments=[augm_strong_fn], **args_dataset_train_s_augm)
		dataset_train_s_augm_strong = FnDataset(dataset_train_s_augm_strong, get_batch_label)

		dataset_train_u_augm_weak = DESEDDataset(augments=[augm_weak_fn], **args_dataset_train_u_augm)
		dataset_train_u_augm_weak = NoLabelDataset(dataset_train_u_augm_weak)

		dataset_train_u_augm_strong = DESEDDataset(augments=[augm_strong_fn], **args_dataset_train_u_augm)
		dataset_train_u_augm_strong = NoLabelDataset(dataset_train_u_augm_strong)

		dataset_train_u_augms_strongs = MultipleDataset([dataset_train_u_augm_strong] * hparams_rmm.nb_augms_strong)
		dataset_train_u_augms_weak_strongs = MultipleDataset([dataset_train_u_augm_weak, dataset_train_u_augms_strongs])

		loader_train_s_augm_strong = DataLoader(dataset=dataset_train_s_augm_strong, **args_loader_train_s)
		loader_train_u_augms_weak_strongs = DataLoader(dataset=dataset_train_u_augms_weak_strongs, **args_loader_train_u)

		train_remixmatch(
			model_factory(), acti_fn, loader_train_s_augm_strong, loader_train_u_augms_weak_strongs, loader_val,
			metrics_s, metrics_u, metrics_u1, metrics_r, metrics_val, hparams_rmm
		)

	if "sf" in args.run:
		hparams_sf = default_supervised_hparams()
		hparams_sf.update(hparams)

		dataset_train_s = DESEDDataset(**args_dataset_train_s)
		dataset_train_s = FnDataset(dataset_train_s, get_batch_label)

		loader_train_s = DataLoader(dataset=dataset_train_s, **args_loader_train_s)

		train_supervised(
			model_factory(), acti_fn, loader_train_s, loader_val, metrics_s, metrics_val,
			hparams_sf, suffix="full_100"
		)

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
