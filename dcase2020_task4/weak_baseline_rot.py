
from torch import nn, Tensor
from dcase2020_task4.baseline.models import WeakBaseline


class WeakBaselineRot(WeakBaseline):
	def __init__(self, nb_rot: int = 4):
		super().__init__()

		self.classifier_rot = nn.Sequential(
			nn.Flatten(),
			nn.Linear(1696, nb_rot)
		)
		self.classifier_count = nn.Sequential(
			nn.Flatten(),
			nn.Linear(1696, 10)
		)

	def forward_rot(self, x: Tensor) -> Tensor:
		x = x.view(-1, 1, *x.shape[1:])

		x = self.features(x)
		x = self.classifier_rot(x)

		return x

	def forward_count(self, x: Tensor) -> Tensor:
		x = x.view(-1, 1, *x.shape[1:])

		x = self.features(x)
		x = self.classifier_count(x)

		return x
