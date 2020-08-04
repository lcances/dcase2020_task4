
from torch.utils.data import Dataset

from dcase2020_task4.util.datasets.fn_dataset import FnDataset
from dcase2020_task4.util.utils_labels import onehot_to_smooth_onehot


class SmoothOneHotDataset(FnDataset):
	"""
		Convert onehot label to smoothed label.
	"""
	def __init__(self, dataset: Dataset, nb_classes: int, smooth: float):
		convert_label_fn = lambda item: (item[0], onehot_to_smooth_onehot(item[1], nb_classes, smooth))
		super().__init__(dataset, fn=convert_label_fn)
		self.nb_classes = nb_classes
