from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from dcase2020_task4.util.dataset_idx import get_classes_idx, shuffle_classes_idx, split_classes_idx
from dcase2020_task4.util.MergeDataLoader import MergeDataLoader
from dcase2020_task4.util.NoLabelDataLoader import NoLabelDataLoader


class DummyDataset(Dataset):
	def __getitem__(self, item):
		return item, 0

	def __len__(self):
		return 10


def test_1():
	ds = DummyDataset()
	nb_classes = 5
	ratios = [0.2, 0.8]

	cls_idx_all = get_classes_idx(ds, nb_classes)
	cls_idx_all = shuffle_classes_idx(cls_idx_all)
	idx_train = split_classes_idx(cls_idx_all, ratios)

	loader_0 = DataLoader(ds, batch_size=2, drop_last=True, sampler=SubsetRandomSampler(idx_train[0]))
	loader_1 = DataLoader(ds, batch_size=3, drop_last=False, sampler=SubsetRandomSampler(idx_train[1]))
	loader = MergeDataLoader([loader_0, loader_1])

	for items in loader:
		print("Items: ", items)


def test_2():
	ds = DummyDataset()

	loader = NoLabelDataLoader(ds)
	for items in loader:
		print("Items: ", items)
		break


if __name__ == "__main__":
	test_1()
	test_2()
