import json
import os
import os.path as osp
import torch

from argparse import ArgumentParser, Namespace
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader

from augmentation_utils.img_augmentations import Transform
from augmentation_utils.signal_augmentations import TimeStretch, Noise, Noise2, Occlusion, PitchShiftRandom
from augmentation_utils.spec_augmentations import RandomTimeDropout, RandomFreqDropout, HorizontalFlip, VerticalFlip
from augmentation_utils.spec_augmentations import Noise as NoiseS

from dcase2020.util.utils import get_datetime, reset_seed

from dcase2020_task4.supervised.trainer import SupervisedTrainer
from dcase2020_task4.validator import ValidatorTag
from dcase2020_task4.util.checkpoint import CheckPoint
from dcase2020_task4.util.fn_dataset import FnDataset
from dcase2020_task4.util.other_augments import RandCropSpec, InversionSpec
from dcase2020_task4.util.other_metrics import CategoricalAccuracyOnehot, MaxMetric, FnMetric, EqConfidenceMetric
from dcase2020_task4.util.utils_match import cross_entropy
from dcase2020_task4.util.utils_standalone import model_factory, optim_factory, sched_factory
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
	parser.add_argument("--dataset_name", type=str, default="UBS8K")
	parser.add_argument("--dataset_path", type=str, default="/projets/samova/leocances/UrbanSound8K/")
	parser.add_argument("--nb_classes", type=int, default=10)
	parser.add_argument("--seed", type=int, default=123)

	parser.add_argument("--batch_size_s", type=int, default=64)
	parser.add_argument("--num_workers_s", type=int, default=4)
	parser.add_argument("--nb_epochs", type=int, default=100)
	parser.add_argument("--checkpoint_path", type=str, default="../models/")
	parser.add_argument("--checkpoint_metric_name", type=str, default="acc")
	parser.add_argument("--confidence", type=float, default=0.5)

	parser.add_argument("--model", type=str, default="CNN03Rot", choices=["UBS8KBaselineRot", "CNN03Rot"])
	parser.add_argument("--optimizer", type=str, default="Adam")
	parser.add_argument("--scheduler", type=str, default=None)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--weight_decay", type=float, default=0.0)
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
	augms_data = [
		(Identity, dict()),
		(Noise, dict(ratio=ratio, target_snr=15.0)),
		(InversionSpec, dict(ratio=ratio)),
		(Noise2, dict(ratio=ratio, noise_factor=(10.0, 10.0))),
		(NoiseSpec, dict(ratio=ratio, snr=15.0)),
		(Occlusion, dict(ratio=ratio, max_size=1.0)),
		(PitchShiftRandom, dict(ratio=ratio, steps=(-1, 1))),
		(RandCropSpec, dict(ratio=ratio, fill_value=-80)),
		(RandomTimeDropout, dict(ratio=ratio, dropout=0.5)),
		(RandomTimeDropout, dict(ratio=ratio, dropout=0.1)),
		(RandomFreqDropout, dict(ratio=ratio, dropout=0.5)),
		(RandomFreqDropout, dict(ratio=ratio, dropout=0.1)),
		(TimeStretch, dict(ratio=ratio)),
		(Transform, dict(ratio=ratio, scale=(0.9, 1.1))),
		(Transform, dict(ratio=ratio, translation=(-10, 10))),
		(Transform, dict(ratio=ratio, scale=(0.5, 1.5))),
		(Transform, dict(ratio=ratio, translation=(-100, 100))),
		(HorizontalFlip, dict(ratio=ratio)),
		(VerticalFlip, dict(ratio=ratio)),
	]

	augms = [cls for cls, _ in augms_data]
	augms_kwargs = [kwargs for _, kwargs in augms_data]

	metadata_root = osp.join(args.dataset_path, "metadata")
	audio_root = osp.join(args.dataset_path, "audio")

	fold_val = 10
	folds_train = list(range(1, 11))
	folds_train.remove(fold_val)
	folds_train = tuple(folds_train)
	folds_val = (fold_val,)

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

	manager = UBS8KDatasetManager(metadata_root, audio_root)
	acti_fn = lambda x, dim: x.softmax(dim=dim).clamp(min=2e-30)
	label_one_hot = lambda item: (item[0], one_hot(torch.as_tensor(item[1]), args.nb_classes).numpy())

	dataset_val_origin = UBS8KDataset(manager, folds=folds_val, augments=(), cached=True)
	dataset_val_origin = FnDataset(dataset_val_origin, label_one_hot)
	loader_val_origin = DataLoader(dataset_val_origin, batch_size=args.batch_size_s, shuffle=False, drop_last=True)

	augm_train = Identity
	augm_train_kwargs = {}

	augm_train_name = get_augm_with_args_name(augm_train, augm_train_kwargs)
	augm_train_fn = augm_train()

	filename = "%s_%d_%d_%s_%s.torch" % (
		args.model, args.nb_epochs, args.batch_size_s, args.checkpoint_metric_name, augm_train_name)
	filename_tmp = filename + ".tmp"

	filepath = osp.join(args.checkpoint_path, filename)
	filepath_tmp = osp.join(args.checkpoint_path, filename_tmp)

	if not osp.isfile(filepath):
		dataset_train = UBS8KDataset(manager, folds=folds_train, augments=(augm_train_fn,), cached=False)
		dataset_train = FnDataset(dataset_train, label_one_hot)
		loader_train = DataLoader(
			dataset_train, batch_size=args.batch_size_s, shuffle=True, num_workers=args.num_workers_s, drop_last=True)

		model = model_factory(args)
		optim = optim_factory(args, model)
		sched = sched_factory(args, optim)

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
		model = model_factory(args)
		model.load_state_dict(state_dict["state_dict"])
		print("Load model \"%s\" with best metric \"%f\"." % (filepath, state_dict["best_metric"]))

	if augm_train_name not in results.keys():
		results[augm_train_name] = {}

	for augm_val, augm_val_kwargs in zip(augms, augms_kwargs):
		augm_val_name = get_augm_with_args_name(augm_val, augm_val_kwargs)
		augm_val_fn = augm_val(**augm_val_kwargs)

		dataset_val = UBS8KDataset(manager, folds=folds_val, augments=(augm_val_fn,), cached=False)
		dataset_val = FnDataset(dataset_val, label_one_hot)
		loader_val = DataLoader(dataset_val, batch_size=args.batch_size_s, shuffle=False, drop_last=True)
		validator = ValidatorTag(
			model, acti_fn, loader_val, metrics_val, None, None, args.checkpoint_metric_name
		)
		validator.val(0)

		_mins, maxs = validator.get_metrics_recorder().get_mins_maxs()
		acc_max = maxs["acc"]
		print("[%s][%s] Acc max = %f" % (augm_train_name, augm_val_name, acc_max))
		results[augm_train_name][augm_val_name] = acc_max

		augm_dic = {get_augm_with_args_name(augm, augm_kwargs): augm_kwargs for augm, augm_kwargs in zip(augms, augms_kwargs)}
		data = {"results": results, "augments": augm_dic, "args": args.__dict__}

		filepath = "results_%s.json" % start_date
		with open(filepath, "w") as file:
			json.dump(data, file, indent="\t")


if __name__ == "__main__":
	main()
