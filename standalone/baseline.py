# dataset manager
from dcase2020.datasetManager import DESEDManager
from dcase2020.datasets import DESEDManager

# utility function and metrics
import dcase2020.augmentation_utils.signal_augmentations as signal_augmentations
import dcase2020.augmentation_utils.spec_augmentations as spec_augmentations
import dcase2020.augmentation_utils.signal_augmentations as signal_augmentations
from dcase2020.pytorch_metrics.metrics import FScore
from dcase2020.util.utils import get_datetime, reset_seed

# ==== set the log ====
import logging
import logging.config
from dcase2020.util.log import DEFAULT_LOGGING
logging.config.dictConfig(DEFAULT_LOGGING)
log = logging.getLogger(__name__)

# ==== reset the seedfor reproductability ====
reset_seed(1234)