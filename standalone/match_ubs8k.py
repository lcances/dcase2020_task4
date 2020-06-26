import os

os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

import torch

from argparse import ArgumentParser, Namespace
from easydict import EasyDict as edict
from time import time

from dcase2020.util.utils import get_datetime, reset_seed
from ubs8k.datasetManager import DatasetManager
from ubs8k.generators import Dataset


def create_args() -> Namespace:
	bool_fn = lambda x: str(x).lower() in ["true", "1", "yes", "y"]

	parser = ArgumentParser()
	parser.add_argument("--run", type=str, default=["supervised", "fixmatch"])
	parser.add_argument("--seed", type=int, default=123)
	parser.add_argument("--debug_mode", "--debug", type=bool_fn, default=False)
	parser.add_argument("--begin_date", type=str, default=get_datetime(),
						help="Date used in SummaryWriter name.")

	return parser.parse_args()


def check_args(args: Namespace):
	pass


def main():
	prog_start = time()
	args = create_args()
	check_args(args)

	reset_seed(args.seed)
	torch.autograd.set_detect_anomaly(args.debug_mode)

	hparams = edict()
	hparams.update(args.__dict__)

	# TODO
	metadata_root = ""
	audio_root = ""
	manager = DatasetManager(metadata_root, audio_root)
	dataset = Dataset(manager, train=True, val=False, augments=[], cached=False)
	# TODO

	exec_time = time() - prog_start
	print("")
	print("Program started at \"%s\" and terminated at \"%s\"." % (hparams.begin_date, get_datetime()))
	print("Total execution time: %.2fs" % exec_time)


if __name__ == "__main__":
	main()
