import numpy as np
import os.path as osp

from argparse import ArgumentParser, Namespace
from easydict import EasyDict as edict
from time import time
from torch.nn.functional import binary_cross_entropy
from torch.utils.data import DataLoader
from torchvision.transforms import RandomChoice, Compose

from dcase2020.augmentation_utils.img_augmentations import Transform
from dcase2020.augmentation_utils.signal_augmentations import TimeStretch, PitchShiftRandom, Noise, Occlusion, Clip
from dcase2020.datasetManager import DESEDManager
from dcase2020.datasets import DESEDDataset

from dcase2020_task4.train_fixmatch import train_fixmatch, default_fixmatch_hparams
from dcase2020_task4.train_mixmatch import train_mixmatch, default_mixmatch_hparams
from dcase2020_task4.train_remixmatch import train_remixmatch, default_remixmatch_hparams
from dcase2020_task4.train_supervised import train_supervised, default_supervised_hparams
from dcase2020_task4.util.FnDataLoader import FnDataLoader
from dcase2020_task4.util.NoLabelDataLoader import NoLabelDataLoader
from dcase2020_task4.util.other_metrics import CategoricalConfidenceAccuracy, FnMetric, MaxMetric
from dcase2020_task4.util.utils import reset_seed, get_datetime
from dcase2020_task4.util.utils_match import to_batch_fn
from dcase2020_task4.weak_baseline_rot import WeakBaselineRot


def create_args() -> Namespace:
	parser = ArgumentParser()
	# TODO : help for acronyms
	parser.add_argument("--run", type=str, nargs="*", default=["fm", "mm", "rmm", "sf"])
	parser.add_argument("--logdir", type=str, default="../../tensorboard")
	parser.add_argument("--dataset", type=str, default="../dataset/DESED")
	parser.add_argument("--seed", type=int, default=123)
	parser.add_argument("--model_name", type=str, default="WeakBaseline", choices=["WeakBaseline"])
	parser.add_argument("--nb_epochs", type=int, default=100)
	parser.add_argument("--batch_size", type=int, default=16)
	parser.add_argument("--nb_classes", type=int, default=10)
	parser.add_argument("--confidence", type=float, default=0.5)
	parser.add_argument("--mode", type=str, default="multihot")
	parser.add_argument("--from_disk", type=bool, default=False,
						help="Select False if you want ot load all data into RAM.")
	return parser.parse_args()


def get_desed_loaders(hparams: edict) -> (DataLoader, DataLoader, DataLoader):
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

	dataset_train_s = DESEDDataset(manager_s, train=True, val=False, augments=[], cached=True, weak=True, strong=False)
	dataset_val = DESEDDataset(manager_s, train=False, val=True, augments=[], cached=True, weak=True, strong=False)
	dataset_train_u = DESEDDataset(manager_u, train=True, val=False, augments=[], cached=True, weak=False, strong=False)

	# Create loaders
	process_fn = lambda batch, labels: (batch, labels[0])
	loader_train_s = FnDataLoader(
		dataset_train_s, batch_size=hparams.batch_size, shuffle=True, num_workers=2, drop_last=True, fn=process_fn)

	loader_val = FnDataLoader(
		dataset_val, batch_size=hparams.batch_size, shuffle=False, num_workers=2, fn=process_fn)

	loader_train_u = NoLabelDataLoader(
		dataset_train_u, batch_size=hparams.batch_size, shuffle=True, num_workers=2, drop_last=True)

	return loader_train_s, loader_train_u, loader_val


def main():
	prog_start = time()

	args = create_args()

	hparams = edict()
	args_filtered = {k: (" ".join(v) if isinstance(v, list) else v) for k, v in args.__dict__.items()}
	hparams.update(args_filtered)
	# Note : some hyperparameters are overwritten when calling the training function, change this in the future
	hparams.begin_date = get_datetime()

	reset_seed(hparams.seed)

	model_factory = lambda: WeakBaselineRot().cuda()
	acti_fn = lambda batch, dim: batch.sigmoid()

	weak_augm_fn = to_batch_fn(RandomChoice([
		Transform(0.5, scale=(0.75, 1.25)),
		Transform(0.5, rotation=(-np.pi, np.pi)),
	]))
	strong_augm_fn = to_batch_fn(Compose([
		RandomChoice([
			Transform(1.0, scale=(0.5, 1.5)),
			Transform(1.0, rotation=(-np.pi, np.pi)),
		]),
		RandomChoice([
			# TimeStretch(1.0),
			PitchShiftRandom(1.0),
			# Noise(1.0),
			Occlusion(1.0),
		]),
	]))
	augment_fn = to_batch_fn(RandomChoice([
		Transform(0.5, scale=(0.75, 1.25)),
		Transform(0.5, rotation=(-np.pi, np.pi)),
		PitchShiftRandom(0.5),
		Occlusion(0.5),
	]))

	metrics_s = CategoricalConfidenceAccuracy(hparams.confidence)
	metrics_u = CategoricalConfidenceAccuracy(hparams.confidence)
	metrics_u1 = CategoricalConfidenceAccuracy(hparams.confidence)
	metrics_r = CategoricalConfidenceAccuracy(hparams.confidence)
	metrics_val_lst = [
		CategoricalConfidenceAccuracy(hparams.confidence),
		MaxMetric(),
		FnMetric(binary_cross_entropy),
	]
	metrics_val_names = ["acc", "max", "loss"]

	loader_train_s, loader_train_u, loader_val = get_desed_loaders(hparams)

	if "fm" in args.run:
		hparams_fm = default_fixmatch_hparams()
		hparams_fm.update(hparams)
		train_fixmatch(
			model_factory(), acti_fn, loader_train_s, loader_train_u, loader_val, weak_augm_fn, strong_augm_fn,
			metrics_s, metrics_u, metrics_val_lst, metrics_val_names, hparams_fm
		)
	if "mm" in args.run:
		hparams_mm = default_mixmatch_hparams()
		hparams_mm.update(hparams)
		train_mixmatch(
			model_factory(), acti_fn, loader_train_s, loader_train_u, loader_val, augment_fn,
			metrics_s, metrics_u, metrics_val_lst, metrics_val_names, hparams_mm
		)
		hparams_mm.criterion_unsupervised = "crossentropy"
		train_mixmatch(
			model_factory(), acti_fn, loader_train_s, loader_train_u, loader_val, augment_fn,
			metrics_s, metrics_u, metrics_val_lst, metrics_val_names, hparams_mm
		)
	if "rmm" in args.run:
		hparams_rmm = default_remixmatch_hparams()
		hparams_rmm.update(hparams)
		train_remixmatch(
			model_factory(), acti_fn, loader_train_s, loader_train_u, loader_val, weak_augm_fn, strong_augm_fn,
			metrics_s, metrics_u, metrics_u1, metrics_r, metrics_val_lst, metrics_val_names, hparams_rmm
		)

	if "sf" in args.run:
		hparams_sf = default_supervised_hparams()
		hparams_sf.update(hparams)
		train_supervised(
			model_factory(), acti_fn, loader_train_s, loader_val, metrics_s, metrics_val_lst, metrics_val_names,
			hparams_sf, suffix="full_100"
		)

	exec_time = time() - prog_start
	print("")
	print("Program started at \"%s\" and terminated at \"%s\"." % (hparams.begin_date, get_datetime()))
	print("Total execution time: %.2fs" % exec_time)


if __name__ == "__main__":
	main()
