from abc import ABC
from torch.nn import Module


class ModuleRot(ABC, Module):
	# TODO : use
	def forward_rot(self, x):
		raise NotImplementedError("Abstract method")
