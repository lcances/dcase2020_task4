from typing import Optional


class RampUp:
	"""
		RampUp class.

		Linearly increase a value from "min_value" to "max_value" each time the method "step()" is called.
		Access the current value with method "value()".
		If nb_steps == 0, the coefficient will be 1.0 and the value will be always the max value expected.
	"""
	def __init__(
		self,
		nb_steps: int,
		max_value: float,
		obj: Optional[object] = None,
		attr_name: Optional[str] = None,
		min_value: float = 0.0,
	):
		self.nb_steps = nb_steps
		self.max_value = max_value
		self.obj = obj
		self.attr_name = attr_name
		self.min_value = min_value

		self.cur_step = 0
		self._check_attributes()
		self.reset()

	def reset(self):
		self.cur_step = 0
		self._update_obj()

	def set_obj(self, obj: Optional[object]):
		self.obj = obj
		self._check_attributes()
		self._update_obj()

	def step(self):
		if self.cur_step < self.nb_steps:
			self.cur_step += 1
			self._update_obj()

	def value(self) -> float:
		return (self.max_value - self.min_value) * self.get_coefficient() + self.min_value

	def get_value(self) -> float:
		return self.value()

	def get_coefficient(self) -> float:
		if self.nb_steps > 0:
			return self.cur_step / self.nb_steps
		else:
			return 1.0

	def _update_obj(self):
		if self.obj is not None:
			self.obj.__setattr__(self.attr_name, self.value())

	def _check_attributes(self):
		if self.obj is not None and not hasattr(self.obj, self.attr_name):
			raise RuntimeError("Use RampUp on attribute \"%s\" but the object \"%s\" do not contains this attribute." % (self.attr_name, obj.__class__.__name__))
