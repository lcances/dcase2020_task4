class RandomChoiceDataset(Dataset):
	def __init__(self, datasets: List[Dataset]):
		self.datasets = datasets

	def __getitem__(self, idx: int):
		return self.fn(self.dataset.__getitem__(idx))

	def __len__(self) -> int:
		return len(self.dataset)