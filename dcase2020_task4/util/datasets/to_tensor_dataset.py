import torch

from torch.utils.data import Dataset
from dcase2020_task4.util.datasets.fn_dataset import FnDataset


class ToTensorDataset(FnDataset):
	def __init__(self, dataset: Dataset):
		super().__init__(dataset, fn=lambda item: tuple([torch.as_tensor(elt.tolist()) for elt in item]))
