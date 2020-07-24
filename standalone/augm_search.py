import json
import os
import os.path as osp
import torch

from argparse import ArgumentParser, Namespace
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose
from typing import Callable, List

from augmentation_utils.img_augmentations import Transform
from augmentation_utils.signal_augmentations import TimeStretch, Noise, Noise2, Occlusion, PitchShiftRandom
from augmentation_utils.spec_augmentations import RandomTimeDropout, RandomFreqDropout, HorizontalFlip, VerticalFlip
from augmentation_utils.spec_augmentations import Noise as NoiseS

from dcase2020.util.utils import get_datetime, reset_seed

from dcase2020_task4.supervised.trainer import SupervisedTrainer
from dcase2020_task4.validator import ValidatorTag
from dcase2020_task4.util.checkpoint import CheckPoint
from dcase2020_task4.util.fn_dataset import FnDataset
from dcase2020_task4.util.onehot_dataset import OneHotDataset
from dcase2020_task4.util.other_img_augments import *
from dcase2020_task4.util.other_spec_augments import CutOutSpec, InversionSpec
from dcase2020_task4.util.other_metrics import CategoricalAccuracyOnehot, MaxMetric, FnMetric, EqConfidenceMetric
from dcase2020_task4.util.utils_match import cross_entropy
from dcase2020_task4.util.utils_standalone import get_model_from_args, get_optim_from_args, get_sched_from_args
from dcase2020_task4.learner import Learner

from ubs8k.datasets import Dataset as UBS8KDataset
from ubs8k.datasetManager import DatasetManager as UBS8KDatasetManager


class NoiseSpec(NoiseS):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)


class Identity:
	def __call__(self, x):
		return x


def create_args() -> Namespace:
	parser = ArgumentParser()
	parser.add_argument("--dataset_name", type=str, default="UBS8K", choices=["UBS8K", "CIFAR10"])
	parser.add_argument("--dataset_path", type=str, default="/projets/samova/leocances/UrbanSound8K/")
	parser.add_argument("--nb_classes", type=int, default=10)
	parser.add_argument("--seed", type=int, default=123)

	parser.add_argument("--batch_size_s", type=int, default=64)
	parser.add_argument("--num_workers_s", type=int, default=4)
	parser.add_argument("--nb_epochs", type=int, default=100)
	parser.add_argument("--checkpoint_path", type=str, default="../models/")
	parser.add_argument("--checkpoint_metric_name", type=str, default="acc")
	parser.add_argument("--confidence", type=float, default=0.5)

	parser.add_argument("--model", type=str, default="CNN03Rot", choices=["WideResNet28Rot", "CNN03Rot"])
	parser.add_argument("--optimizer", type=str, default="Adam")
	parser.add_argument("--scheduler", type=str, default=None)
	parser.add_argument("--lr", "--learning_rate", type=float, default=1e-3)
	parser.add_argument("--weight_decay", type=float, default=0.0)

	parser.add_argument("--fold_val", type=int, default=10)

	return parser.parse_args()


def get_augm_with_args_name(augm, augm_kwargs: dict) -> str:
	filter_ = lambda s: str(s)\
		.replace("(", "_op_").replace(")", "_cp_")\
		.replace("[", "_ob_").replace("]", "_cb_")\
		.replace(" ", "_").replace(",", "_c_")
	kwargs_suffix = "_".join([("%s_%s" % (key, filter_(value))) for key, value in sorted(augm_kwargs.items())])
	return "%s_%s" % (augm.__name__, kwargs_suffix)


def main():
	start_date = get_datetime()
	args = create_args()
	reset_seed(args.seed)

	ratio = 1.0
	if args.dataset_name == "UBS8K":
		augms_data = [
			(Identity, dict()),
			(HorizontalFlip, dict(ratio=ratio)),
			(VerticalFlip, dict(ratio=ratio)),
			(Noise, dict(ratio=ratio, target_snr=15)),
			(Noise, dict(ratio=ratio, target_snr=20)),
			(InversionSpec, dict(ratio=ratio)),
			(Noise2, dict(ratio=ratio, noise_factor=(1.0, 1.0))),
			(Noise2, dict(ratio=ratio, noise_factor=(0.5, 0.5))),
			(Noise2, dict(ratio=ratio, noise_factor=(0.1, 0.1))),
			(NoiseSpec, dict(ratio=ratio, snr=15.0)),
			(NoiseSpec, dict(ratio=ratio, snr=20.0)),
			(Occlusion, dict(ratio=ratio, max_size=1.0)),
			(PitchShiftRandom, dict(ratio=ratio, steps=(-1, 1))),
			(PitchShiftRandom, dict(ratio=ratio, steps=(-3, 3))),
			(CutOutSpec, dict(ratio=ratio, fill_value=-80, rect_width_scale_range=(0.1, 0.5), rect_height_scale_range=(0.1, 0.5))),
			(CutOutSpec, dict(ratio=ratio, fill_value=-80, rect_width_scale_range=(0.1, 0.25), rect_height_scale_range=(0.1, 0.25))),
			(RandomTimeDropout, dict(ratio=ratio, dropout=0.5)),
			(RandomTimeDropout, dict(ratio=ratio, dropout=0.01)),
			(RandomFreqDropout, dict(ratio=ratio, dropout=0.5)),
			(RandomFreqDropout, dict(ratio=ratio, dropout=0.01)),
			(TimeStretch, dict(ratio=ratio)),
		]
		"""
			(Transform, dict(ratio=ratio, scale=(0.9, 1.1))),
			(Transform, dict(ratio=ratio, translation=(-10, 10))),
			(Transform, dict(ratio=ratio, scale=(0.5, 1.5))),
			(Transform, dict(ratio=ratio, translation=(-100, 100))),
		"""
	elif args.dataset_name == "CIFAR10":
		ratio = 1.0

		enhance_range = (0.05, 0.95)
		transforms_range = (-0.3, 0.3)
		posterize_range = (4, 8)
		angles_range = (-30, 30)
		thresholds_range = (0, 256)

		augms_data = [
			(Identity, dict()),
			(AutoContrast, dict(ratio=ratio)),
			(Brightness, dict(ratio=ratio, levels=enhance_range)),
			(Color, dict(ratio=ratio, levels=enhance_range)),
			(Contrast, dict(ratio=ratio, levels=enhance_range)),
			(Equalize, dict(ratio=ratio)),
			(Posterize, dict(ratio=ratio, nbs_bits=posterize_range)),
			(Rotation, dict(ratio=ratio, angles=angles_range)),
			(Sharpness, dict(ratio=ratio, levels=enhance_range)),
			(ShearX, dict(ratio=ratio, shears=transforms_range)),
			(ShearY, dict(ratio=ratio, shears=transforms_range)),
			(Solarize, dict(ratio=ratio, thresholds=thresholds_range)),
			(TranslateX, dict(ratio=ratio, deltas=transforms_range)),
			(TranslateY, dict(ratio=ratio, deltas=transforms_range)),
			(Invert, dict(ratio=ratio)),
			(Rescale, dict(ratio=ratio, scales=(0.5, 2.0))),
			(Smooth, dict(ratio=ratio)),
			(HorizontalFlip, dict(ratio=ratio)),
			(VerticalFlip, dict(ratio=ratio)),
			(CutOut, dict(ratio=ratio, rect_width_scale_range=(0.1, 0.5), rect_height_scale_range=(0.1, 0.5))),
		]
	else:
		raise RuntimeError("Unknown dataset %s" % args.dataset_name)

	augms_cls = [cls for cls, _ in augms_data]
	augms_kwargs = [kwargs for _, kwargs in augms_data]
	augms_fn = [cls(**kwargs) for cls, kwargs in augms_data]

	metrics_s = {
		"s_acc": CategoricalAccuracyOnehot(dim=1),
	}
	metrics_val = {
		"acc": CategoricalAccuracyOnehot(dim=1),
		"ce": FnMetric(cross_entropy),
		"eq": EqConfidenceMetric(args.confidence, dim=1),
		"max": MaxMetric(dim=1),
	}
	results = {}

	acti_fn = lambda x, dim: x.softmax(dim=dim).clamp(min=2e-30)

	augm_train_cls = Identity
	augm_train_kwargs = {}

	augm_train_name = get_augm_with_args_name(augm_train_cls, augm_train_kwargs)
	augm_train_fn = augm_train_cls()

	filename = "%s_%d_%d_%s_%s.torch" % (
		args.model, args.nb_epochs, args.batch_size_s, args.checkpoint_metric_name, augm_train_name)
	filename_tmp = filename + ".tmp"

	filepath = osp.join(args.checkpoint_path, filename)
	filepath_tmp = osp.join(args.checkpoint_path, filename_tmp)

	if args.dataset_name == "UBS8K":
		dataset_train, dataset_val_origin, datasets_val = get_ubs8k_datasets(args, augm_train_fn, augms_fn)
	elif args.dataset_name == "CIFAR10":
		dataset_train, dataset_val_origin, datasets_val = get_cifar10_datasets(args, augm_train_fn, augms_fn)
	else:
		raise RuntimeError("Unknown dataset %s" % args.dataset_name)

	dataset_train = OneHotDataset(dataset_train, args.nb_classes)
	dataset_val_origin = OneHotDataset(dataset_val_origin, args.nb_classes)
	for i, _ in enumerate(datasets_val):
		datasets_val[i] = OneHotDataset(datasets_val[i], args.nb_classes)

	if not osp.isfile(filepath):
		loader_train = DataLoader(
			dataset_train, batch_size=args.batch_size_s, shuffle=True, num_workers=args.num_workers_s, drop_last=True)
		loader_val_origin = DataLoader(
			dataset_val_origin, batch_size=args.batch_size_s, shuffle=False, drop_last=True)

		model = get_model_from_args(args)
		optim = get_optim_from_args(args, model)
		sched = get_sched_from_args(args, optim)

		criterion = cross_entropy

		trainer = SupervisedTrainer(
			model, acti_fn, optim, loader_train, metrics_s, criterion, None
		)
		checkpoint = CheckPoint(model, optim, name=filepath_tmp)
		validator = ValidatorTag(
			model, acti_fn, loader_val_origin, metrics_val, None, checkpoint, args.checkpoint_metric_name
		)
		steppables = []
		if sched is not None:
			steppables.append(sched)
		learner = Learner("Supervised_%s" % augm_train_name, trainer, validator, args.nb_epochs, steppables)
		learner.start()

		validator.get_metrics_recorder().print_min_max()

		print("Rename \"%s\" to \"%s\"..." % (filepath_tmp, filepath))
		os.rename(filepath_tmp, filepath)

	else:
		state_dict = torch.load(filepath)
		model = get_model_from_args(args)
		model.load_state_dict(state_dict["state_dict"])
		print("Load model \"%s\" with best metric \"%f\"." % (filepath, state_dict["best_metric"]))

	if augm_train_name not in results.keys():
		results[augm_train_name] = {}

	for augm_val, augm_val_kwargs, dataset_val in zip(augms_cls, augms_kwargs, datasets_val):
		augm_val_name = get_augm_with_args_name(augm_val, augm_val_kwargs)

		loader_val = DataLoader(dataset_val, batch_size=args.batch_size_s, shuffle=False, drop_last=True)
		validator = ValidatorTag(
			model, acti_fn, loader_val, metrics_val, None, None, args.checkpoint_metric_name
		)
		validator.val(0)

		recorder = validator.get_metrics_recorder()
		_, maxs = recorder.get_mins_maxs()
		acc_max = maxs["acc"]
		print("[%s][%s] Acc max = %f" % (augm_train_name, augm_val_name, acc_max))
		results[augm_train_name][augm_val_name] = acc_max

		augm_dic = {
			get_augm_with_args_name(augm, augm_kwargs): augm_kwargs for augm, augm_kwargs in zip(augms_cls, augms_kwargs)
		}
		data = {"results": results, "augments": augm_dic, "args": args.__dict__}

		filepath = "results_%s.json" % start_date
		with open(filepath, "w") as file:
			json.dump(data, file, indent="\t")


def get_cifar10_datasets(
	args: Namespace, augm_train_fn: Callable, augms_val_fn: List[Callable]
):
	pre_process_fn = lambda img: np.array(img)
	post_process_fn = lambda img: img.transpose()

	# Prepare data
	dataset_train = CIFAR10(
		args.dataset_path, train=True, download=True, transform=Compose([pre_process_fn, augm_train_fn, post_process_fn]))

	dataset_val_origin = CIFAR10(
		args.dataset_path, train=False, download=True, transform=Compose([pre_process_fn, post_process_fn]))

	datasets_val = []
	for augm_val_fn in augms_val_fn:
		dataset_val = CIFAR10(
			args.dataset_path, train=False, download=True, transform=Compose([pre_process_fn, augm_val_fn, post_process_fn]))
		datasets_val.append(dataset_val)

	return dataset_train, dataset_val_origin, datasets_val


def get_ubs8k_datasets(
	args: Namespace, augm_train_fn: Callable, augms_val_fn: List[Callable]
) -> (Dataset, Dataset, List[Dataset]):
	metadata_root = osp.join(args.dataset_path, "metadata")
	audio_root = osp.join(args.dataset_path, "audio")

	folds_train = list(range(1, 11))
	folds_train.remove(args.fold_val)
	folds_train = tuple(folds_train)
	folds_val = (args.fold_val,)

	manager = UBS8KDatasetManager(metadata_root, audio_root)

	to_tensor = lambda item: (torch.from_numpy(item[0]), torch.from_numpy(item[1]))

	dataset_train = UBS8KDataset(manager, folds=folds_train, augments=(augm_train_fn,), cached=False)
	dataset_train = FnDataset(dataset_train, to_tensor)

	dataset_val_origin = UBS8KDataset(manager, folds=folds_val, augments=(), cached=True)
	dataset_val_origin = FnDataset(dataset_val_origin, to_tensor)

	datasets_val = []
	for augm_val_fn in augms_val_fn:
		dataset_val = UBS8KDataset(manager, folds=folds_val, augments=(augm_val_fn,), cached=False)
		dataset_val = FnDataset(dataset_val, to_tensor)
		datasets_val.append(dataset_val)

	return dataset_train, dataset_val_origin, datasets_val


if __name__ == "__main__":
	main()
