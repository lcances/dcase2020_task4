# %% Import
import time

import torch.nn as nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

# dataset manager
from dcase2020.datasetManager import DESEDManager
from dcase2020.datasets import DESEDDataset

# utility function & metrics & augmentation
import dcase2020.augmentation_utils.signal_augmentations as signal_augmentations
import dcase2020.augmentation_utils.spec_augmentations as spec_augmentations
import dcase2020.augmentation_utils.signal_augmentations as signal_augmentations
from dcase2020.pytorch_metrics.metrics import FScore
from dcase2020.util.utils import get_datetime, reset_seed

# models
from dcase2020.models import WeakBaseline

# %%
# ==== set the log ====
import logging
import logging.config
from dcase2020.util.log import DEFAULT_LOGGING

logging.config.dictConfig(DEFAULT_LOGGING)
log = logging.getLogger(__name__)

# %%
# ==== reset the seed for reproductability ====
reset_seed(1234)

# %%
# ==== load the dataset ====
dese_metadata_root = "../dataset/DESED/metadata"
desed_audio_root = "../dataset/DESED/audio"

manager = DESEDManager(
    dese_metadata_root, desed_audio_root,
    sampling_rate=22050,
    validation_ratio=0.2,
    verbose=2
)

manager.add_subset("weak")

manager.split_train_validation()

# %%  setup augmentation and create pytorch dataset
augments = [
    # signal_augmentation.Noise(0.5, target_snr=15),
    # signal_augmentation.RandomTimeDropout(0.5, dropout=0.2)
]

train_dataset = DESEDDataset(manager, train=True, val=False, augments=augments, cached=True)
val_dataset = DESEDDataset(manager, train=False, val=True, augments=[], cached=True)

# %% Setup model and training parameters

model = WeakBaseline()

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
fscore_func = FScore()


# %% Training

# %% training function
def train(epoch: int):
    start_time = time.time()
    fscore_func.reset()
    model.train()
    print("")  # <-- Force new line

    for i, (X, y) in enumerate(training_loader):
        X, y = X.cuda().float(), y.cuda().long()

        logits = model(X)

        loss = criterion(logits, y)

        # calc metrics
        _, pred = torch.max(logits, 1)
        f1 = fscore_func(pred, y)

        # back propagation
        optimizers.zero_grad()
        loss.backward()
        optimizers.step()

        # logs
        print("Epoch {}, {:d}% \t loss: {:.4e} - f1: {:.4e} - took {:.2f}s".format(
            epoch + 1,
            int(100 * (i + 1) / nb_batch),
            loss.item(),
            f1,
            time.time() - start_time
        ), end="\r")

    # tensorboard logs
    tensorboard.add_scalar("train/loss", loss.item(), epoch)
    tensorboard.add_scalar("train/f1", f1, epoch)


# %% validation function
def val(epoch):
    fscore_func.reset()
    model.train()
    print("")  # <-- Force new line

    for i, (X, y) in enumerate(val_loader):
        X, y = X.cuda().float(), y.cuda().long()

        logits = model(X)

        loss = criterion(logits, y)

        # calc metrics
        _, pred = torch.max(logits, 1)
        f1 = fscore_func(pred, y)

        # back propagation
        optimizers.zero_grad()
        loss.backward()
        optimizers.step()

        # logs
        print("validation \t val_loss: {:.4e} - val_f1: {:.4e}".format(
            loss.item(),
            f1,
        ), end="\r")

    # tensorboard logs
    tensorboard.add_scalar("val/loss", loss.item(), epoch)
    tensorboard.add_scalar("val/f1", f1, epoch)


# %%
for e in range(nb_epochs):
    train()
    val()