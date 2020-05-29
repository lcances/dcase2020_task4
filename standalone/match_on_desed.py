import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# dataset manager
from dcase2020.datasetManager import DESEDManager
from dcase2020.datasets import DESEDDataset

# utility function & metrics & augmentation
from dcase2020_task4.util.utils import get_datetime, reset_seed

# models
from dcase2020_task4.baseline.models import WeakBaseline


# ==== set the log ====
import logging.config
from dcase2020_task4.util.log import DEFAULT_LOGGING
logging.config.dictConfig(DEFAULT_LOGGING)
log = logging.getLogger(__name__)


def get_desed_loaders(desed_metadata_root: str, desed_audio_root: str) -> (DataLoader, DataLoader):
	manager = DESEDManager(
		desed_metadata_root, desed_audio_root,
		sampling_rate=22050,
		validation_ratio=0.2,
		verbose=2
	)

	manager.add_subset("weak")
	manager.add_subset("unlabel_in_domain")
	manager.split_train_validation()

	train_dataset = DESEDDataset(manager, train=True, val=False, augments=[], cached=True)
	val_dataset = DESEDDataset(manager, train=False, val=True, augments=[], cached=True)

	batch_size = 16

	# loaders
	training_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

	return training_loader, val_loader


def main():
	# ==== reset the seed for reproducibility ====
	reset_seed(1234)

	# ==== load the dataset ====
	desed_metadata_root = "../../dataset/DESED/dataset/metadata"
	desed_audio_root = "../../dataset/DESED/dataset/audio"

	loader_train, loader_val = get_desed_loaders(desed_metadata_root, desed_audio_root)
	print("TODO")


if __name__ == "__main__":
	main()
