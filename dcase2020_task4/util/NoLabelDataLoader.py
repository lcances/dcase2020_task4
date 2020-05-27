from torch.utils.data import DataLoader


class NoLabelDataLoader(DataLoader):
	def __iter__(self):
		for x, _y in super().__iter__():
			yield [x]
