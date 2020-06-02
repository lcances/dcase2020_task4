
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

		self.current_value = min_value

	def reset(self):
		self.current_value = self.min_value

	def step(self) -> float:
		if self.current_value < self.max_value:
			self.current_value += self.max_value / self.nb_steps
		return self.current_value

	def value(self) -> float:
		return self.current_value
