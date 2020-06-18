
class RampUp:
	"""
		RampUp class.

		Linearly increase a value from "min_value" to "max_value" each time the method "step()" is called.
		Access the current value with property "value".
	"""
	def __init__(self, max_value: float, nb_steps: int, min_value: float = 0.0):
		self.max_value = max_value
		self.nb_steps = nb_steps
		self.min_value = min_value

		self.cur_step = 0

	def reset(self):
		self.cur_step = 0

	def step(self):
		if self.cur_step < self.nb_steps:
			self.cur_step += 1

	def value(self) -> float:
		return (self.max_value - self.min_value) * self.get_coef() + self.min_value

	def get_coef(self) -> float:
		return self.cur_step / self.nb_steps


def test():
	rampup = RampUp(1.0, 10)
	for i in range(15):
		print("Value:", rampup.value())
		rampup.step()


if __name__ == "__main__":
	test()
