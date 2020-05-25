from torch.utils.data import DataLoader


class NoLabelDataLoader(DataLoader):
	def __iter__(self):
		x, _y = super().__iter__()
		return x
