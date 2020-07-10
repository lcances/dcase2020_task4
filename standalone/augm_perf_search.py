import json
import os.path as osp
import torch

from argparse import ArgumentParser, Namespace
from torch.utils.data import DataLoader

from augmentation_utils.spec_augmentations import RandomTimeDropout, RandomFreqDropout, Noise, HorizontalFlip, VerticalFlip

from dcase2020_task4.supervised.trainer import SupervisedTrainer
from dcase2020_task4.validator import DefaultValidator
from dcase2020_task4.util.other_augments import Gray, Inversion, RandCrop, UniColor, RandCropSpec
from dcase2020_task4.util.other_metrics import CategoricalAccuracyOnehot, MaxMetric, FnMetric, EqConfidenceMetric
from dcase2020_task4.util.checkpoint import CheckPoint
from dcase2020_task4.util.utils_match import cross_entropy
from dcase2020_task4.util.utils_standalone import model_factory, optim_factory
from dcase2020_task4.learner import DefaultLearner

from ubs8k.datasets import Dataset as UBS8KDataset
from ubs8k.datasetManager import DatasetManager as UBS8KDatasetManager


class Identity:
	def __call__(self, x):
		return x


def create_args() -> Namespace:
	parser = ArgumentParser()
	parser.add_argument("--dataset", type=str, default="/projets/samova/leocances/UrbanSound8K/")
	parser.add_argument("--batch_size_s", type=int, default=64)
	parser.add_argument("--num_workers_s", type=int, default=4)
	parser.add_argument("--nb_epochs", type=int, default=100)
	parser.add_argument("--path_checkpoint", type=str, default="~/root/task4/models/")
	parser.add_argument("--checkpoint_metric_name", type=str, default="acc")
	return parser.parse_args()


def main():
	args = create_args()

	ratio = 1.0
	augms = [
		Identity,
		RandomTimeDropout,
		# RandomFreqDropout,
		# Noise,
		# HorizontalFlip,
		# VerticalFlip,
	]
	augms_kwargs = [
		dict(ratio=ratio),
		dict(ratio=ratio),
		dict(ratio=ratio),
		dict(ratio=ratio, snr=15.0),
		dict(ratio=ratio),
		dict(ratio=ratio),
	]

	metadata_root = osp.join(args.dataset, "metadata")
	audio_root = osp.join(args.dataset, "audio")

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

	dataset_val_origin = UBS8KDataset(manager, folds=folds_val, augments=(), cached=True)
	loader_val_origin = DataLoader(dataset_val_origin, batch_size=args.batch_size_s, shuffle=False, drop_last=True)

	for i, (augm_train, augm_train_kwargs) in enumerate(zip(augms, augms_kwargs)):
		augm_train_name = augm_train.__name__
		augm_train_fn = augm_train(**augm_train_kwargs)

		kwargs_suffix = "_".join([value for key, value in sorted(augm_train_kwargs.items())])
		filename = "%s_%s_%d_%d_%s_%s.torch" % (
			args.model_name, augm_train_name, args.nb_epochs, args.batch_size_s, args.checkpoint_metric_name, kwargs_suffix)
		filepath = osp.join(args.path_checkpoint, filename)

		if not osp.isfile(filepath):
			dataset_train = UBS8KDataset(manager, folds=folds_train, augments=(augm_train_fn,), cached=False)
			loader_train = DataLoader(dataset_train, batch_size=args.batch_size_s, shuffle=True, num_workers=args.num_workers_s, drop_last=True)

			model = model_factory(args)
			optim = optim_factory(args, model)

			criterion = cross_entropy

			trainer = SupervisedTrainer(
				model, acti_fn, optim, loader_train, metrics_s, criterion, None
			)
			checkpoint = CheckPoint(model, optim, name=filepath)
			validator = DefaultValidator(
				model, acti_fn, loader_val_origin, metrics_val, None, checkpoint, args.checkpoint_metric_name
			)
			learner = DefaultLearner("Supervised_%s" % augm_train_name, trainer, validator, args.nb_epochs)
			learner.start()

			validator.get_metrics_recorder().print_min_max()

		else:
			state_dict = torch.load(filepath)
			model = model_factory(args)
			model.load_state_dict(state_dict["state_dict"])
			print("Load model \"%s\" with best metric \"%f\"." % (filepath, state_dict["best_metric"]))

		if augm_train_name not in results.keys():
			results[augm_train_name] = {}

		for j, (augm_val, augm_val_kwargs) in enumerate(zip(augms, augms_kwargs)):
			augm_val_name = augm_val.__name__
			augm_val_fn = augm_val(**augm_val_kwargs)

			dataset_val = UBS8KDataset(manager, folds=folds_val, augments=(augm_val_fn,), cached=False)
			loader_val = DataLoader(dataset_val, batch_size=args.batch_size_s, shuffle=False, drop_last=True)
			validator = DefaultValidator(
				model, acti_fn, loader_val, metrics_val, None, None, args.checkpoint_metric_name
			)

			_mins, maxs = validator.get_metrics_recorder().get_mins_maxs()
			acc_max = maxs["acc"]
			print("[%s][%s] Acc max = %f" % (augm_train_name, augm_val_name, acc_max))
			results[augm_train_name][augm_val_name] = acc_max

			augm_dic = {augm.__name__: augm_kwargs for augm, augm_kwargs in zip(augms, augms_kwargs)}
			data = {"results": results, "augments": augm_dic}
			json.dump(data, open("results.json", "w"))


if __name__ == "__main__":
	main()
