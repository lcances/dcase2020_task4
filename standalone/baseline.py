import time

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

# dataset manager
from dcase2020.datasetManager import DESEDManager
from dcase2020.datasets import DESEDDataset

# utility function & metrics & augmentation
import dcase2020_task4.augmentation_utils.signal_augmentations as signal_augmentations
import dcase2020_task4.augmentation_utils.spec_augmentations as spec_augmentations
import dcase2020_task4.augmentation_utils.img_augmentations as img_augmentations
from dcase2020_task4.pytorch_metrics.metrics import FScore, BinaryAccuracy
from dcase2020_task4.util.utils import get_datetime, reset_seed

# models
from dcase2020_task4.baseline.models import WeakBaseline


# ==== set the log ====
import logging.config
from dcase2020_task4.util.log import DEFAULT_LOGGING
logging.config.dictConfig(DEFAULT_LOGGING)
log = logging.getLogger(__name__)

# ==== reset the seed for reproductability ====
reset_seed(1234)

# ==== load the dataset ====
dese_metadata_root = "../dataset/DESED/dataset/metadata"
desed_audio_root = "../dataset/DESED/dataset/audio"

manager = DESEDManager(
    dese_metadata_root, desed_audio_root,
    sampling_rate=22050,
    validation_ratio=0.2,
    verbose=2
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

    for i, (X, y) in enumerate(training_loader):
        X, y = X.cuda().float(), y.cuda().float()

        logits = model(X)

        loss = criterion(logits, y)

        # calc metrics
        pred = F.sigmoid(logits)
        binacc = binacc_func(pred, y)

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

    for i, (X, y) in enumerate(val_loader):
        X, y = X.cuda().float(), y.cuda().float()

        logits = model(X)

        loss = criterion(logits, y)

        # calc metrics
        pred = F.sigmoid(logits)
        binacc = binacc_func(pred, y)

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
