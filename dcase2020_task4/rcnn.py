import torch

from torch import Tensor, nn
from torch.nn import Sequential
from torch.nn.modules import Module


class ConvBNReluMax(Sequential):
	def __init__(self, in_size: int):
		super().__init__(
			nn.Conv2d(in_size, 64, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=(4, 1)),
			nn.Dropout2d(0.3)  # TODO : SpatialDropout ?
		)


if __name__ == "__main__":
	x = torch.zeros(8, 64, 431)
	x = x.view(-1, 1, *x.shape[1:])

	print("x = ", x.shape)
	x = ConvBNReluMax(1)(x)
	print("x = ", x.shape)
	x = ConvBNReluMax(64)(x)
	print("x = ", x.shape)
	x = ConvBNReluMax(64)(x)
	print("x = ", x.shape)

	x = x.squeeze(dim=2)
	print("x = ", x.shape)
	x, y = nn.GRU(input_size=431, hidden_size=64, bidirectional=True)(x)
	print("gru = ", x.shape)
	# print(y.shape)
	x = nn.Flatten()(x)
	print("x = ", x.shape)
	x = nn.Linear(8192, 64)(x)
	x = nn.ReLU()(x)
	print("x = ", x.shape)
	x = nn.Linear(64, 10)(x)
	x = nn.Sigmoid()(x)
	print("loc_out = ", x.shape)

	x = x.view(-1, 1, *x.shape[1:])
	x1, x2 = nn.AvgPool1d(1)(x), nn.MaxPool1d(1)(x)
	x1, x2 = nn.Flatten()(x1), nn.Flatten()(x2)
	print("x1 = ", x1.shape)
	print("x2 = ", x2.shape)
	x = torch.cat((x1, x2), dim=1)
	print("x = ", x.shape)
	x = nn.Linear(20, 1024)(x)
	x = nn.ReLU()(x)
	x = nn.Linear(1024, 10)(x)
	x = nn.Sigmoid()(x)
	print("at_out = ", x.shape)


class RCNN(Module):
	def __init__(self):
		super().__init__()

		self.features = nn.Sequential(
			ConvBNReluMax(1),
			ConvBNReluMax(64),
			ConvBNReluMax(64),
			# nn.Flatten(),  # Reshape ?
			nn.GRU(input_size=64, hidden_size=64, bidirectional=True),
			nn.Tanh(),
			nn.Dropout(0.1),
		)

		self.classifier_loc_output = nn.Sequential(
			nn.Linear(64, 1024),
			nn.ReLU(),
			nn.Linear(1024, 10),
			nn.Sigmoid(),
		)

	def forward(self, x: Tensor) -> Tensor:
		print("In: ", x.shape)
		x = x.view(-1, 1, *x.shape[1:])
		print("View: ", x.shape)
		x = self.features(x)
		print("Feat: ", x.shape)
		x = self.classifier(x)
		print("Clas: ", x.shape)
		return x
