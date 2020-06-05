
from torch.utils.data import Dataset
from dcase2020_task4.util.FnDataset import FnDataset


class NoLabelDataset(FnDataset):
	def __init__(self, dataset: Dataset):
		super().__init__(dataset, fn=lambda item: item[0])
