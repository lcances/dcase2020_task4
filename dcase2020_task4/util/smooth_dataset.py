
from torch.utils.data import Dataset

from dcase2020_task4.util.fn_dataset import FnDataset
from dcase2020_task4.util.utils_labels import onehot_to_smooth_onehot


class SmoothOneHotDataset(FnDataset):
	def __init__(self, dataset: Dataset, nb_classes: int, smooth: float):
		convert_label_fn = lambda item: tuple(item[:-1]) + (onehot_to_smooth_onehot(item[-1], nb_classes, smooth),)
		super().__init__(dataset, fn=convert_label_fn)
		self.nb_classes = nb_classes
