import numpy as np

from typing import Callable, List, Optional, Tuple, Union


class UniLoss:
	def __init__(
		self,
		attributes: Union[List[Tuple[object, str]], List[Tuple[object, str, Callable]]],
		ratios_range: List[Tuple[List[float], int, int]],
	):
		self.attributes = attributes
		self.ratios_range = ratios_range

		self.cur_step = 0
		self.default_value = 0.0

		for i, tuple_ in enumerate(self.attributes):
			if len(tuple_) == 2:
				obj, attr_name = tuple_
				value = obj.__getattribute__(attr_name)
				self.attributes[i] = (obj, attr_name, lambda: value)

		self.reset()

	def reset(self):
		self.cur_step = 0
		self.default_value = 0.0

		self._choose_loss()

	def step(self):
		self._choose_loss()
		self.cur_step += 1

	def _choose_loss(self):
		for ratios, epoch_min, epoch_max in self.ratios_range:
			if epoch_min <= self.cur_step <= epoch_max:
				cur_loss_idx = np.random.choice(range(len(ratios)), p=ratios)

				for i, (obj, attr_name, value_fn) in enumerate(self.attributes):
					new_value = value_fn() if i == cur_loss_idx else self.default_value
					obj.__setattr__(attr_name, new_value)
				break


def test():
	class A:
		def __init__(self):
			self.a = 1
			self.b = 2
			self.c = 3

	class DummyRampup:
		def __init__(self, value: float, nb: int):
			self.value = value
			self.nb = nb
			self.i = 0

		def step(self):
			if self.i < self.nb:
				self.i += 1

		def __call__(self) -> float:
			return self.value * self.i / self.nb

	obj = A()
	ramp = DummyRampup(15.0, 5)

	uni_loss = UniLoss(
		attributes=[(obj, "a", ramp), (obj, "b"), (obj, "c")],
		ratios_range=[
			([0.5, 0.5, 0.0], 0, 9),
			([0.5, 0.5, 0.0], 10, 14),
			([0.05, 0.9, 0.05], 15, 20),
		]
	)

	for i in range(20):
		print("[%2d] obj.a : %.2f ; obj.b : %.2f ; obj.c : %.2f" % (i, obj.a, obj.b, obj.c))
		ramp.step()
		uni_loss.step()


if __name__ == "__main__":
	test()
