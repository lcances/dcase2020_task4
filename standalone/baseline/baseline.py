import time
import os
import argparse
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append("..")

# dataset manager
from dcase2020.datasetManager import DESEDManager
from dcase2020.datasets import DESEDDataset

# utility function & metrics & augmentation
from metric_utils.metrics import FScore, BinaryAccuracy
from dcase2020_task4.util.utils import get_datetime, reset_seed

# models
from dcase2020_task4.baseline.models import WeakBaseline

# ==== set the log ====
import logging.config
from dcase2020_task4.util.log import DEFAULT_LOGGING
logging.config.dictConfig(DEFAULT_LOGGING)
log = logging.getLogger(__name__)

# ==== Get the arguments ====
parser = argparse.ArgumentParser()

parser.add_argument("-a", "--audio_root", default="../../dataset/DESED/dataset/audio", type=str)
parser.add_argument("-m", "--metadata_root", default="../../dataset/DESED/dataset/metadata", type=str)

args = parser.parse_args()

# ==== reset the seed for reproducibility ====
reset_seed(1234)

# ==== load the dataset ====
desed_metadata_root = args.metadata_root
desed_audio_root = args.audio_root

manager = DESEDManager(
    desed_metadata_root, desed_audio_root,
    sampling_rate=22050,
    validation_ratio=0.2,
    from_disk=False,
    verbose=1
)

manager.add_subset("weak")

manager.split_train_validation()

# setup augmentation and create pytorch dataset
augments = [
    # signal_augmentation.Noise(0.5, target_snr=15),
    # signal_augmentation.RandomTimeDropout(0.5, dropout=0.2)
]

train_dataset = DESEDDataset(manager, train=True, val=False, augments=augments, cached=True)
val_dataset = DESEDDataset(manager, train=False, val=True, augments=[], cached=True)

# ======== Prepare training ========
model = WeakBaseline()
model.cuda()

# training parameters
nb_epochs = 100
batch_size = 32
nb_batch = len(train_dataset) // batch_size

# criterion & optimizers
criterion = nn.BCEWithLogitsLoss(reduction="mean")

optimizers = torch.optim.Adam(model.parameters(), lr=0.003)

# callbacks
callbacks = []

# tensorboard
title = "WeakBaseline_%s" % (get_datetime())
tensorboard = SummaryWriter(log_dir="../tensorboard/%s" % title, comment="weak baseline")

# loaders
training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Metrics
binacc_func = BinaryAccuracy()


# Training functions
def train(epoch: int):
    start_time = time.time()
    binacc_func.reset()
    model.train()
    print("")  # <-- Force new line

    for i, (X_weak, y_weak) in enumerate(training_loader):
        # The DESEDDataset return a list of ground truth depending on the selecting option.
        # If weak and strong ground truth are selected, the list order is [WEAK, STRONG]
        # here there is only one [WEAK]
        X_weak, y_weak = X_weak.cuda().float(), y_weak[0].cuda().float()

        logits = model(X_weak)

        loss = criterion(logits, y_weak)

        # calc metrics
        pred = F.sigmoid(logits)
        binacc = binacc_func(pred, y_weak)

        # back propagation
        optimizers.zero_grad()
        loss.backward()
        optimizers.step()

        # logs
        print("Epoch {}, {:d}% \t loss: {:.4e} - acc: {:.4e} - took {:.2f}s".format(
            epoch + 1,
            int(100 * (i + 1) / nb_batch),
            loss.item(),
            binacc,
            time.time() - start_time
        ), end="\r")

    # tensorboard logs
    tensorboard.add_scalar("train/loss", loss.item(), epoch)
    tensorboard.add_scalar("train/acc", binacc, epoch)


def val(epoch):
    binacc_func.reset()
    model.train()
    print("")  # <-- Force new line

    for i, (X_weak, y_weak) in enumerate(val_loader):
        X_weak, y_weak = X_weak.cuda().float(), y_weak[0].cuda().float()

        logits = model(X_weak)

        loss = criterion(logits, y_weak)

        # calc metrics
        pred = F.sigmoid(logits)
        binacc = binacc_func(pred, y_weak)

        # back propagation
        optimizers.zero_grad()
        loss.backward()
        optimizers.step()

        # logs
        print("validation \t val_loss: {:.4e} - val_acc: {:.4e}".format(
            loss.item(),
            binacc,
        ), end="\r")

    # tensorboard logs
    tensorboard.add_scalar("val/loss", loss.item(), epoch)
    tensorboard.add_scalar("val/acc", binacc, epoch)


# Train
for e in range(nb_epochs):
    train(e)
    val(e)

# ♫♪.ılılıll|̲̅̅●̲̅̅|̲̅̅=̲̅̅|̲̅̅●̲̅̅|llılılı.♫♪
