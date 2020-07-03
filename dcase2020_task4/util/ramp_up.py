from typing import Optional


class RampUp:
	"""
		RampUp class.

		Linearly increase a value from "min_value" to "max_value" each time the method "step()" is called.
		Access the current value with method "value()".
		If nb_steps == 0, the coefficient will be 1.0 and the value will be always the max value expected.
	"""
	def __init__(self, nb_steps: int, max_value: Optional[float] = None, min_value: float = 0.0):
		self.nb_steps = nb_steps
		self.max_value = max_value
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
		if self.nb_steps > 0:
			return self.cur_step / self.nb_steps
		else:
			return 1.0
