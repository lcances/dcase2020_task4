from dcase2020_task4.util.ZipLongestCycle import ZipLongestCycle


def test():
	r1 = range(1, 4)
	r2 = range(1, 6)
	iters = ZipLongestCycle([r1, r2])
	for v1, v2 in iters:
		print(v1, v2)


if __name__ == "__main__":
	test()
