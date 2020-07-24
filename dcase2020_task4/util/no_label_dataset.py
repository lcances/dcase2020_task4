
from torch.utils.data import Dataset
from dcase2020_task4.util.fn_dataset import FnDataset


class NoLabelDataset(FnDataset):
	"""
		Remove label from dataset by getting only the batch.
	"""
	def __init__(self, dataset: Dataset):
		super().__init__(dataset, fn=lambda item: item[0])
