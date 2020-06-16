
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

		self.step = 0

	def reset(self):
		self.step = 0

	def step(self):
		if self.step < self.nb_steps:
			self.step += 1

	def value(self) -> float:
		return (self.max_value - self.min_value) * self.get_coef() + self.min_value

	def get_coef(self) -> float:
		return self.step / self.nb_steps
