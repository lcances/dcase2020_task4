from dcase2020_task4.util.uniloss import ConstantEpochUniloss


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
			return self.value * self.i / self.nb + 10

	obj = A()
	ramp = DummyRampup(15.0, 5)

	nb_epochs = 100
	begin_s = 1
	begin_unif = int(nb_epochs * 0.1)
	begin_u = int(nb_epochs * 0.9)

	print("Ranges: [%d, %d]" % (begin_s, begin_unif - 1))
	print("Ranges: [%d, %d]" % (begin_unif, begin_u - 1))
	print("Ranges: [%d, %d]" % (begin_u, nb_epochs))

	uni_loss = ConstantEpochUniloss(
		attributes=[(obj, "a", ramp), (obj, "b")],
		ratios_range=[
			([1.0, 0.0], begin_s, begin_unif - 1),
			([1.0, 0.0], begin_unif, begin_u - 1),
			([0.0, 1.0], begin_u, nb_epochs),
		]
	)

	for i in range(nb_epochs):
		print("[%2d] obj.a : %.2f ; obj.b : %.2f ; obj.c : %.2f" % (i+1, obj.a, obj.b, obj.c))
		ramp.step()
		uni_loss.step()


if __name__ == "__main__":
	test()
