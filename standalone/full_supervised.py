
import os

os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
import time
import logging

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append("../src/")

from datasetManager import DESEDManager
from datasets import DESEDDataset

from utils import get_datetime, get_model_from_name, reset_seed, set_logs
from metrics import CategoricalAccuracy

import augmentation_utils.signal_augmentations
import augmentation_utils.img_augmentations
import augmentation_utils.spec_augmentations

# Arguments ========
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--subset", action="append", help="subset to use for training: [weak | unlabel_in_domain | synthetic]")
parser.add_argument("--validation_ratio", default=0.2, type=float, help="percentage of subset to use for validation")

#parser.add_argument("--subsampling", default=1.0, type=float, help="subsampling ratio")
#parser.add_argument("--subsampling_method", default="balance", type=str, help="subsampling method [random|balance]")

parser.add_argument("--seed", default=1234, type=int, help="Seed for random generation. Use for reproductability")
parser.add_argument("--model", default="cnn", type=str, help="Model to load, see list of model in models.py")
parser.add_argument("-T", "--log_dir", default="Test", required=True, help="Tensorboard working directory")
parser.add_argument("-j", "--job_name", default="default")
parser.add_argument("--log", default="warning", help="Log level")
parser.add_argument("-a","--augments", action="append", help="Augmentation. use as if python script" )
parser.add_argument("--num_workers", default=4, type=int, help="Choose numver of worker to train the model")

args = parser.parse_args()

# Logging system
set_logs(args.log)

# Reproducibility
reset_seed(args.seed)

# load the data
# TODO most likely to change due to bad implementation of the validation system
metadata_root="../dataset/DESED/dataset/metadata"
audio_root="../dataset/DESED/dataset/audio"

manager = DESEDManager(
    metadata_root,
    audio_root,
    22050,
    validation_ratio=0.2,
    verbose=1
)

for subset in args.subset:
    manager.add_subset(subset)

manager.split_train_validation()

# Create the model

# create the pytorch dataset
train_dataset = DESEDDataset(manager, train=True, val=False, augments=[], cached=True)
val_dataset = DESEDDataset(manager, train=False, val=True, augments=[], cached=True)