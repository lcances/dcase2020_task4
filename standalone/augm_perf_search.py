import json
import numpy as np
import os.path as osp

from argparse import ArgumentParser, Namespace
from torch.utils.data import DataLoader

from dcase2020_task4.supervised.trainer import SupervisedTrainer
from dcase2020_task4.validator import DefaultValidator
from dcase2020_task4.util.other_augments import Gray, Inversion, RandCrop, UniColor, RandCropSpec
from dcase2020_task4.util.other_metrics import CategoricalAccuracyOnehot, MaxMetric, FnMetric, EqConfidenceMetric
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
	return parser.parse_args()


def main():
	args = create_args()

	metadata_root = osp.join(args.dataset, "metadata")
	audio_root = osp.join(args.dataset, "audio")

	augms = [Identity()]

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
	results = np.zeros(len(augms), len(augms))

	manager = UBS8KDatasetManager(metadata_root, audio_root)
	acti_fn = lambda x, dim: x.softmax(dim=dim).clamp(min=2e-30)

	dataset_val_origin = UBS8KDataset(manager, folds=folds_val, augments=(), cached=True)
	loader_val_origin = DataLoader(dataset_val_origin, batch_size=args.batch_size_s, shuffle=False, drop_last=True)

	for i, augm_train in enumerate(augms):
		dataset_train = UBS8KDataset(manager, folds=folds_train, augments=(augm_train,), cached=False)
		loader_train = DataLoader(dataset_train, batch_size=args.batch_size_s, shuffle=True, num_workers=args.num_workers_s, drop_last=True)

		model = model_factory(args)
		optim = optim_factory(args, model)

		criterion = cross_entropy

		trainer = SupervisedTrainer(
			model, acti_fn, optim, loader_train, metrics_s, criterion, None
		)

		validator = DefaultValidator(
			model, acti_fn, loader_val_origin, metrics_val, None, None, args.checkpoint_metric_name
		)
		learner = DefaultLearner("Supervised", trainer, validator, args.nb_epochs)
		learner.start()

		validator.get_metrics_recorder().print_min_max()

		for j, augm_val in enumerate(augms):
			dataset_val = UBS8KDataset(manager, folds=folds_val, augments=(augm_val,), cached=False)
			loader_val = DataLoader(dataset_val, batch_size=args.batch_size_s, shuffle=False, drop_last=True)
			validator = DefaultValidator(
				model, acti_fn, loader_val, metrics_val, None, None, args.checkpoint_metric_name
			)
			mins, maxs = validator.get_metrics_recorder().get_mins_maxs()
			acc_max = maxs["acc"]
			print("[%d][%d] Acc max = %f" % (i, j, acc_max))
			results[i][j] = acc_max

			data = {"results": results, "augments": [augm.__class__.__name__ for augm in augms]}
			json.dump(data, open("results.json", "w"))


if __name__ == "__main__":
	main()
