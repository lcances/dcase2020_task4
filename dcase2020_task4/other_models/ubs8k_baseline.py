from torch import nn
from dcase2020_task4.baseline.layers import ConvPoolReLU


class UBS8KBaseline(nn.Module):
	def __init__(self, **kwargs):
		nn.Module.__init__(self)

		self.features = nn.Sequential(
			ConvPoolReLU(1, 32, 3, 1, 1, pool_kernel_size=(4, 2), pool_stride=(4, 2), dropout=0.0),
			ConvPoolReLU(32, 32, 3, 1, 1, pool_kernel_size=(4, 2), pool_stride=(4, 2), dropout=0.3),
			ConvPoolReLU(32, 32, 3, 1, 1, pool_kernel_size=(4, 2), pool_stride=(4, 2), dropout=0.3),
			nn.Conv2d(32, 32, 1, 1, 0),
			nn.ReLU6(),
		)

		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Dropout(0.5),
			nn.Linear(672, 10)
		)

	def forward(self, x):
		x = x.view(-1, 1, *x.shape[1:])

		x = self.features(x)
		x = self.classifier(x)

		return x


class UBS8KBaselineRot(UBS8KBaseline):
	def __init__(self, rot_output_size: int = 4, **kwargs):
		super().__init__(**kwargs)
		self.classifier_rot = nn.Sequential(
			nn.Flatten(),
			nn.Dropout(0.5),
			nn.Linear(672, rot_output_size)
		)

	def forward_rot(self, x):
		x = x.view(-1, 1, *x.shape[1:])

		x = self.features(x)
		x = self.classifier_rot(x)

		return x