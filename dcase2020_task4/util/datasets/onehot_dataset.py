
from torch.utils.data import Dataset

from dcase2020_task4.util.datasets.fn_dataset import FnDataset
from dcase2020_task4.util.utils_labels import nums_to_onehot


class OneHotDataset(FnDataset):
	"""
		Convert index label to onehot label.
	"""
	def __init__(self, dataset: Dataset, nb_classes: int):
		convert_label_fn = lambda item: (item[0], nums_to_onehot(item[1], nb_classes))
		super().__init__(dataset, fn=convert_label_fn)
		self.nb_classes = nb_classes
