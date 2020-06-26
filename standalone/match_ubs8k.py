import os

os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

import os.path as osp
import torch

from argparse import ArgumentParser, Namespace
from time import time

from dcase2020.util.utils import get_datetime, reset_seed
from ubs8k.datasetManager import DatasetManager
from ubs8k.datasets import Dataset


def create_args() -> Namespace:
	bool_fn = lambda x: str(x).lower() in ["true", "1", "yes", "y"]

	parser = ArgumentParser()
	parser.add_argument("--run", type=str, default=["fixmatch"])
	parser.add_argument("--seed", type=int, default=123)
	parser.add_argument("--debug_mode", "--debug", type=bool_fn, default=False)
	parser.add_argument("--begin_date", type=str, default=get_datetime(),
						help="Date used in SummaryWriter name.")

	parser.add_argument("--mode", type=str, default="multihot")
	parser.add_argument("--dataset", type=str, default="/projets/samova/leocances/UrbanSound8K")
	parser.add_argument("--dataset_name", type=str, default="UbS8K")
	parser.add_argument("--logdir", type=str, default="../../tensorboard/")

	return parser.parse_args()


def check_args(args: Namespace):
	if not osp.isdir(args.dataset):
		raise RuntimeError("Invalid dirpath %s" % args.dataset)

	if args.write_results:
		if not osp.isdir(args.logdir):
			raise RuntimeError("Invalid dirpath %s" % args.logdir)


def main():
	start_time = time()
	start_date = get_datetime()

	args = create_args()
	check_args(args)

	reset_seed(args.seed)
	torch.autograd.set_detect_anomaly(args.debug_mode)

	# TODO
	metadata_root = osp.join(args.dataset, "metadata")
	audio_root = osp.join(args.dataset, "audio")
	manager = DatasetManager(metadata_root, audio_root)
	dataset_train = Dataset(manager, folds=(1, 2, 3, 4, 5, 6, 7, 8, 9), augments=[], cached=False)
	dataset_val = Dataset(manager, folds=(10,), augments=[], cached=True)
	# TODO

	# TODO

	exec_time = time() - start_time
	print("")
	print("Program started at \"%s\" and terminated at \"%s\"." % (start_date, get_datetime()))
	print("Total execution time: %.2fs" % exec_time)


if __name__ == "__main__":
	main()
