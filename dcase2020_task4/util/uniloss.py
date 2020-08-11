import numpy as np

from typing import Callable, List, Tuple, Union


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
		self._choose_loss()

	def step(self):
		self.cur_step += 1
		self._choose_loss()

	def _choose_loss(self):
		for ratios, epoch_min, epoch_max in self.ratios_range:
			if epoch_min <= self.cur_step <= epoch_max:
				cur_loss_idx = np.random.choice(range(len(ratios)), p=ratios)

				for i, (obj, attr_name, value_fn) in enumerate(self.attributes):
					new_value = value_fn() if i == cur_loss_idx else self.default_value
					obj.__setattr__(attr_name, new_value)
				break


class WeightLinearUniLoss:
	def __init__(self, targets: List[Tuple[object, str, float, float, float]], nb_steps: int):
		"""
			@param targets: List of tuples (object to update, attribute name, constant value, probability at start, probability at end)
			@param nb_steps: Nb of steps max. Can be the number of iterations multiply by the number of epochs.
		"""
		self.targets = targets
		self.nb_steps = nb_steps
		self.index_step = 0

		self._update_objects()

	def reset(self):
		self.index_step = 0

	def step(self):
		self.index_step += 1
		self._update_objects()

	def get_current_ratios(self) -> List[float]:
		ratios = []
		for _, _, _, ratio_start, ratio_end in self.targets:
			ratio = self.index_step / self.nb_steps * (ratio_end - ratio_start) + ratio_start
			ratios.append(ratio)
		return ratios

	def _update_objects(self):
		ratios = self.get_current_ratios()
		chosen = np.random.choice(range(len(self.targets)), p=ratios)

		for i, (obj, attr_name, value, _, _) in enumerate(self.targets):
			cur_value = value if i == chosen else 0.0
			obj.__setattr__(attr_name, cur_value)
