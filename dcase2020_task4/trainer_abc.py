
from abc import ABC
from typing import Dict, List
from metric_utils.metrics import Metrics


class TrainerABC(ABC):
	def train(self, epoch: int):
		raise NotImplementedError("Abstract method")

	def get_all_metrics(self) -> List[Dict[str, Metrics]]:
		raise NotImplementedError("Abstract method")

	def get_nb_iterations(self) -> int:
		raise NotImplementedError("Abstract method")

	def reset_all_metrics(self):
		all_metrics = self.get_all_metrics()
		for metrics in all_metrics:
			for metric in metrics.values():
				metric.reset()
