import torch

import logging
import logging.config
from dcase2020.util.log import DEFAULT_LOGGING
logging.config.dictConfig(DEFAULT_LOGGING)
log = logging.getLogger(__name__)


class CheckPoint:
    def __init__(self, model, optimizer, mode: str="max", name: str="best"):
        self.mode = mode
        self.name = name

        self.model = model
        self.optimizer = optimizer

        self.best_state = None
        self.best_metric = 0 if mode == "max" else 100000
        self.epoch_counter = 0

    def step(self, new_value):
        self.epoch_counter += 1

        if self._check_is_better(new_value):
            print("")
            log.info("Best performance reached: saving the model")
            
            self.best_metric = new_value
            self.best_state = {
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_metric": self.best_metric,
                "epoch": self.epoch_counter,
            }
            torch.save(self.best_state, self.name)

    def _check_is_better(self, new_value):
        if self.mode == "max":
            if self.best_metric <= new_value:
                return True
            return False

        if self.best_metric <= new_value:
            return False
        return True
