import numpy as np

from argparse import ArgumentParser, Namespace
from easydict import EasyDict as edict
from time import time
from torch.nn import Module
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import CIFAR10

from dcase2020.util.utils import get_datetime, reset_seed

from dcase2020_task4.util.MergeDataLoader import MergeDataLoader
from dcase2020_task4.util.dataset_idx import get_classes_idx, shuffle_classes_idx, reduce_classes_idx, split_classes_idx
from dcase2020_task4.resnet import ResNet18
from dcase2020_task4.vgg import VGG

from dcase2020_task4.train_fixmatch import test_fixmatch
from dcase2020_task4.train_mixmatch import test_mixmatch
from dcase2020_task4.train_remixmatch import test_remixmatch
from dcase2020_task4.train_supervised import test_supervised


def get_nb_parameters(model: Module) -> int:
	return sum(p.numel() for p in model.parameters())


def get_nb_trainable_parameters(model: Module) -> int:
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_args() -> Namespace:
	parser = ArgumentParser()
	parser.add_argument("--run", type=str, nargs="*", default=["fm", "mm", "rmm", "full", "part"])
	parser.add_argument("--dataset", type=str, default="dataset/CIFAR10")
	parser.add_argument("--logdir", type=str, default="../tensorboard")
	parser.add_argument("--seed", type=int, default=123)
	parser.add_argument("--model_name", type=str, default="VGG11")
	parser.add_argument("--nb_epochs", type=int, default=100)
	parser.add_argument("--dataset_ratio", type=float, default=1.0)
	parser.add_argument("--supervised_ratio", type=float, default=0.1)
	return parser.parse_args()


def main():
	prog_start = time()

	args = create_args()

	hparams = edict()
	hparams.update(args.__dict__)

	# Note : some hyperparameters are overwritten when calling the training function
	hparams.begin_date = get_datetime()
	hparams.batch_size = 16
	hparams.nb_classes = 10
	hparams.confidence = 0.3

	reset_seed(hparams.seed)

	# Prepare data
	transform_fn = lambda img: np.array(img).transpose()  # Transpose img [3, 32, 32] to [32, 32, 3]
	train_set = CIFAR10(hparams.dataset, train=True, download=True, transform=transform_fn)
	val_set = CIFAR10(hparams.dataset, train=False, download=True, transform=transform_fn)

	# Create loaders
	sub_loaders_ratios = [hparams.supervised_ratio, 1.0 - hparams.supervised_ratio]

	cls_idx_all = get_classes_idx(train_set, hparams.nb_classes)
	cls_idx_all = shuffle_classes_idx(cls_idx_all)
	cls_idx_all = reduce_classes_idx(cls_idx_all, hparams.dataset_ratio)

	idx_train = split_classes_idx(cls_idx_all, sub_loaders_ratios)
	idx_train_s, idx_train_u = idx_train
	idx_val = list(range(int(len(val_set) * hparams.dataset_ratio)))

	loader_train_full = DataLoader(
		train_set, batch_size=hparams.batch_size, sampler=SubsetRandomSampler(idx_train_s + idx_train_u), num_workers=2,
		drop_last=True
	)
	loader_train_s = DataLoader(
		train_set, batch_size=hparams.batch_size, sampler=SubsetRandomSampler(idx_train_s), num_workers=2, drop_last=True
	)
	loader_train_u = DataLoader(
		train_set, batch_size=hparams.batch_size, sampler=SubsetRandomSampler(idx_train_u), num_workers=2, drop_last=True
	)
	loader_train_ss = MergeDataLoader([loader_train_s, loader_train_u])

	loader_val = DataLoader(
		val_set, batch_size=hparams.batch_size, sampler=SubsetRandomSampler(idx_val), num_workers=2
	)

	# Create model
	if hparams.model_name == "VGG11":
		model_factory = lambda: VGG("VGG11").cuda()
	elif hparams.model_name == "ResNet18":
		model_factory = lambda: ResNet18().cuda()
	else:
		raise RuntimeError("Unknown model %s" % hparams.model_name)

	print("Model selected : %s (%d parameters, %d trainable parameters)." % (
		hparams.model_name, get_nb_parameters(model_factory()), get_nb_trainable_parameters(model_factory())))

	if "fm" in args.run:
		test_fixmatch(model_factory(), loader_train_ss, loader_val, edict(hparams))
	if "mm" in args.run:
		test_mixmatch(model_factory(), loader_train_ss, loader_val, edict(hparams))
	if "rmm" in args.run:
		test_remixmatch(model_factory(), loader_train_ss, loader_val, edict(hparams))

	if "full" in args.run:
		test_supervised(model_factory(), loader_train_full, loader_val, edict(hparams), suffix="full_100")
	if "part" in args.run:
		test_supervised(model_factory(), loader_train_s, loader_val, edict(hparams), suffix="part_%d" % int(100 * hparams.supervised_ratio))

	exec_time = time() - prog_start
	print("Total execution time: %.2fs" % exec_time)


if __name__ == "__main__":
	main()
