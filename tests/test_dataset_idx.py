from dcase2020_task4.util.datasets.dataset_idx import split_classes_idx


def test() -> bool:
	tests = [
		([[1, 2], [3, 4], [5, 6]], [0.5, 0.5]),
		([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [0.5, 0.25, 0.25]),
	]
	expected_lst = [
		[[1, 3, 5], [2, 4, 6]],
		[[1, 2, 5, 6, 9, 10], [3, 7, 11], [4, 8, 12]],
	]
	ok = True
	for (indices, ratios), expected in zip(tests, expected_lst):
		c = split_classes_idx(indices, ratios)
		if c == expected:
			print("OK")
		else:
			print("KO")
			ok = False

	return ok


if __name__ == "__main__":
	test()
