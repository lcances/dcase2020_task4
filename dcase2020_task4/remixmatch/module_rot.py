from abc import ABC
from torch.nn import Module


class ModuleRot(ABC, Module):
	def forward_rot(self, x):
		raise NotImplementedError("Abstract method")
