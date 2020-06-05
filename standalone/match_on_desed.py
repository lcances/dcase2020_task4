import os
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

import numpy as np
import os.path as osp

from argparse import ArgumentParser, Namespace
from easydict import EasyDict as edict
from time import time
from torch.nn.functional import binary_cross_entropy
from torch.utils.data import DataLoader
from torchvision.transforms import RandomChoice, Compose

from dcase2020.augmentation_utils.img_augmentations import Transform
from dcase2020.augmentation_utils.signal_augmentations import TimeStretch, PitchShiftRandom, Noise, Occlusion
from dcase2020.datasetManager import DESEDManager
from dcase2020.datasets import DESEDDataset

from dcase2020_task4.train_fixmatch import train_fixmatch, default_fixmatch_hparams
from dcase2020_task4.train_mixmatch import train_mixmatch, default_mixmatch_hparams
from dcase2020_task4.train_remixmatch import train_remixmatch, default_remixmatch_hparams
from dcase2020_task4.train_supervised import train_supervised, default_supervised_hparams
from dcase2020_task4.util.FnDataset import FnDataset
from dcase2020_task4.util.MultipleDataset import MultipleDataset
from dcase2020_task4.util.NoLabelDataset import NoLabelDataset
from dcase2020_task4.util.other_metrics import BinaryConfidenceAccuracy, FnMetric, MaxMetric, EqConfidenceMetric
from dcase2020_task4.util.utils import reset_seed, get_datetime
from dcase2020_task4.weak_baseline_rot import WeakBaselineRot


def create_args() -> Namespace:
	parser = ArgumentParser()
	# TODO : help for acronyms
	parser.add_argument("--run", type=str, nargs="*", default=["fm", "mm", "rmm", "sf"], choices=["fm", "mm", "rmm", "sf"])
	parser.add_argument("--logdir", type=str, default="../../tensorboard")
	parser.add_argument("--dataset", type=str, default="../dataset/DESED")
	parser.add_argument("--seed", type=int, default=123)
	parser.add_argument("--model_name", type=str, default="WeakBaseline", choices=["WeakBaseline"])
	parser.add_argument("--nb_epochs", type=int, default=100)
	parser.add_argument("--batch_size", type=int, default=64)
	parser.add_argument("--nb_classes", type=int, default=10)
	parser.add_argument("--confidence", type=float, default=0.5)
	parser.add_argument("--mode", type=str, default="multihot")
	parser.add_argument("--from_disk", type=bool, default=True,
						help="Select False if you want ot load all data into RAM.")
	parser.add_argument("--num_workers_s", type=int, default=4)
	parser.add_argument("--num_workers_u", type=int, default=4)

	parser.add_argument("--nb_augms", type=int, default=2, help="Nb of augmentations used in MixMatch.")
	parser.add_argument("--nb_augms_strong", type=int, default=2, help="Nb of strong augmentations used in ReMixMatch.")
	return parser.parse_args()


def get_desed_managers(hparams: edict) -> (DESEDManager, DESEDManager):
	desed_metadata_root = osp.join(hparams.dataset, osp.join("dataset", "metadata"))
	desed_audio_root = osp.join(hparams.dataset, osp.join("dataset", "audio"))

	manager_s = DESEDManager(
		desed_metadata_root, desed_audio_root,
		from_disk=hparams.from_disk,
		sampling_rate=22050,
		validation_ratio=0.2,
		verbose=1
	)
	manager_s.add_subset("weak")
	# manager_s.add_subset("synthetic20")
	manager_s.split_train_validation()

	manager_u = DESEDManager(
		desed_metadata_root, desed_audio_root,
		from_disk=hparams.from_disk,
		sampling_rate=22050,
		validation_ratio=0.0,
		verbose=1
	)
	manager_u.add_subset("unlabel_in_domain")
	manager_u.split_train_validation()

	return manager_s, manager_u


def main():
	prog_start = time()

	args = create_args()

	hparams = edict()
	args_filtered = {k: (" ".join(v) if isinstance(v, list) else v) for k, v in args.__dict__.items()}
	hparams.update(args_filtered)
	# Note : some hyperparameters are overwritten when calling the training function, change this in the future
	hparams.begin_date = get_datetime()
	hparams.dataset_name = "DESED"

	reset_seed(hparams.seed)

	model_factory = lambda: WeakBaselineRot().cuda()
	acti_fn = lambda batch, dim: batch.sigmoid()

	weak_augm_fn = RandomChoice([
		Transform(0.5, scale=(0.75, 1.25)),
		Transform(0.5, rotation=(-np.pi, np.pi)),
	])
	strong_augm_fn = Compose([
		RandomChoice([
			Transform(1.0, scale=(0.5, 1.5)),
			Transform(1.0, rotation=(-np.pi, np.pi)),
		]),
		RandomChoice([
			TimeStretch(1.0),
			PitchShiftRandom(1.0),
			Noise(1.0),
			Occlusion(1.0),
		]),
	])
	augment_fn = RandomChoice([
		Transform(0.5, scale=(0.75, 1.25)),
		Transform(0.5, rotation=(-np.pi, np.pi)),
		PitchShiftRandom(0.5),
		Occlusion(0.5),
	])

	metric_s = BinaryConfidenceAccuracy(hparams.confidence)
	metric_u = BinaryConfidenceAccuracy(hparams.confidence)
	metric_u1 = BinaryConfidenceAccuracy(hparams.confidence)
	metric_r = BinaryConfidenceAccuracy(hparams.confidence)
	metrics_val = {
		"acc": BinaryConfidenceAccuracy(hparams.confidence),
		"max": MaxMetric(),
		"loss": FnMetric(binary_cross_entropy),
		"eq": EqConfidenceMetric(hparams.confidence)
	}

	manager_s, manager_u = get_desed_managers(hparams)

	# Validation
	get_batch_label = lambda item: (item[0], item[1][0])
	dataset_val = DESEDDataset(manager_s, train=False, val=True, augments=[], cached=True, weak=True, strong=False)
	dataset_val = FnDataset(dataset_val, get_batch_label)
	loader_val = DataLoader(dataset_val, batch_size=hparams.batch_size, shuffle=False)

	if "fm" in args.run:
		hparams_fm = default_fixmatch_hparams()
		hparams_fm.update(hparams)

		dataset_train_s_weak = DESEDDataset(
			manager_s, train=True, val=False, augments=[weak_augm_fn], cached=False, weak=True, strong=False)
		dataset_train_s_weak = FnDataset(dataset_train_s_weak, get_batch_label)

		dataset_train_u_weak = DESEDDataset(
			manager_u, train=True, val=False, augments=[weak_augm_fn], cached=False, weak=False, strong=False)
		dataset_train_u_weak = NoLabelDataset(dataset_train_u_weak)

		dataset_train_u_strong = DESEDDataset(
			manager_u, train=True, val=False, augments=[strong_augm_fn], cached=False, weak=False, strong=False)
		dataset_train_u_strong = NoLabelDataset(dataset_train_u_strong)

		dataset_train_u_weak_strong = MultipleDataset([dataset_train_u_weak, dataset_train_u_strong])

		loader_train_s_weak = DataLoader(
			dataset_train_s_weak, batch_size=hparams.batch_size, shuffle=True, num_workers=hparams.num_workers_s,
			drop_last=True)
		loader_train_u_weak_strong = DataLoader(
			dataset_train_u_weak_strong, batch_size=hparams.batch_size, shuffle=True, num_workers=hparams.num_workers_u,
			drop_last=True)

		train_fixmatch(
			model_factory(), acti_fn, loader_train_s_weak, loader_train_u_weak_strong, loader_val,
			metric_s, metric_u, metrics_val, hparams_fm
		)

	if "mm" in args.run:
		hparams_mm = default_mixmatch_hparams()
		hparams_mm.update(hparams)

		dataset_train_s_augm = DESEDDataset(manager_s, train=True, val=False, augments=[augment_fn], cached=False,
											weak=True, strong=False)
		dataset_train_s_augm = FnDataset(dataset_train_s_augm, get_batch_label)

		dataset_train_u_augm = DESEDDataset(manager_u, train=True, val=False, augments=[augment_fn], cached=False,
											weak=False, strong=False)
		dataset_train_u_augm = NoLabelDataset(dataset_train_u_augm)

		dataset_train_u_augms = MultipleDataset([dataset_train_u_augm] * hparams.nb_augms)

		loader_train_s_augm = DataLoader(
			dataset_train_s_augm, batch_size=hparams.batch_size, shuffle=True, num_workers=hparams.num_workers_s,
			drop_last=True)
		loader_train_u_augms = DataLoader(
			dataset_train_u_augms, batch_size=hparams.batch_size, shuffle=True, num_workers=hparams.num_workers_u,
			drop_last=True)

		train_mixmatch(
			model_factory(), acti_fn, loader_train_s_augm, loader_train_u_augms, loader_val,
			metric_s, metric_u, metrics_val, hparams_mm
		)
		hparams_mm.criterion_unsupervised = "crossentropy"
		train_mixmatch(
			model_factory(), acti_fn, loader_train_s_augm, loader_train_u_augms, loader_val,
			metric_s, metric_u, metrics_val, hparams_mm
		)
	if "rmm" in args.run:
		hparams_rmm = default_remixmatch_hparams()
		hparams_rmm.update(hparams)

		# ReMixMatch
		dataset_train_s_strong = DESEDDataset(manager_s, train=True, val=False, augments=[strong_augm_fn], cached=False,
											  weak=True, strong=False)
		dataset_train_u_weak = DESEDDataset(manager_u, train=True, val=False, augments=[weak_augm_fn], cached=False,
											weak=False, strong=False)
		dataset_train_u_strong = DESEDDataset(manager_u, train=True, val=False, augments=[strong_augm_fn], cached=False,
											  weak=False, strong=False)
		dataset_train_u_strongs = MultipleDataset([dataset_train_u_strong] * hparams.nb_augms_strong)
		dataset_train_u_weak_strongs = MultipleDataset([dataset_train_u_weak, dataset_train_u_strongs])

		loader_train_s_strong = DataLoader(
			dataset_train_s_strong, batch_size=hparams.batch_size, shuffle=True, num_workers=hparams.num_workers_s,
			drop_last=True)
		loader_train_u_weak_strongs = DataLoader(
			dataset_train_u_weak_strongs, batch_size=hparams.batch_size, shuffle=True,
			num_workers=hparams.num_workers_u, drop_last=True)

		train_remixmatch(
			model_factory(), acti_fn, loader_train_s_strong, loader_train_u_weak_strongs, loader_val,
			metric_s, metric_u, metric_u1, metric_r, metrics_val, hparams_rmm
		)
	if "sf" in args.run:
		hparams_sf = default_supervised_hparams()
		hparams_sf.update(hparams)

		dataset_train_s = DESEDDataset(
			manager_s, train=True, val=False, augments=[], cached=True, weak=True, strong=False)
		dataset_train_s = FnDataset(dataset_train_s, get_batch_label)
		loader_train_s = DataLoader(
			dataset_train_s, batch_size=hparams.batch_size, shuffle=True, num_workers=hparams.num_workers_s, drop_last=True)

		train_supervised(
			model_factory(), acti_fn, loader_train_s, loader_val, metric_s, metrics_val,
			hparams_sf, suffix="full_100"
		)

	exec_time = time() - prog_start
	print("")
	print("Program started at \"%s\" and terminated at \"%s\"." % (hparams.begin_date, get_datetime()))
	print("Total execution time: %.2fs" % exec_time)


if __name__ == "__main__":
	main()
