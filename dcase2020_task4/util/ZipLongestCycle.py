
from typing import Iterable, Sized


class ZipLongestCycle(Iterable, Sized):
	"""
		Merge iteration of collections.

		Example :
		r1 = range(1, 3)
		r2 = range(1, 5)
		multiter = MultipleIterable([r1, r2])
		for v1, v2 in multiter:
			print(v1, v2)

		will print :
		1 1
		2 2
		3 3
		1 4
		2 5
	"""

	def __init__(self, iterables: list):
		self._iterables = iterables

		# Check iterable size for avoid exception in __iter__
		for iterable in self._iterables:
			if len(iterable) == 0:
				raise RuntimeError("An iterable is empty.")

		self._len = max([len(iterable) for iterable in self._iterables])

	def __iter__(self) -> list:
		cur_iters = [iter(iterable) for iterable in self._iterables]
		cur_count = [0 for _ in self._iterables]

		for _ in range(len(self)):
			items = []

			for i, _ in enumerate(cur_iters):
				if cur_count[i] < len(self._iterables[i]):
					item = next(cur_iters[i])
					cur_count[i] += 1
				else:
					cur_iters[i] = iter(self._iterables[i])
					item = next(cur_iters[i])
					cur_count[i] = 1
				items.append(item)

			yield items

	def __len__(self) -> int:
		return self._len
