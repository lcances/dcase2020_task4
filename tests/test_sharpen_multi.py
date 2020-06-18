import torch

from dcase2020_task4.util.utils_match import sharpen_multi, sharpen_multi_1, sharpen_multi_2


def test():
	distribution = torch.as_tensor([
		0.9, 0.4, 0.6
	])
	print("Distribution:", distribution)

	result = sharpen_multi_2(distribution, 0.5, 0.5)
	print("Result:", result)


if __name__ == "__main__":
	test()
