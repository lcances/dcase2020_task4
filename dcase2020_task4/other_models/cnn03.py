from torch import nn
from torch.nn import Module, Sequential

from dcase2020_task4.baseline.layers import ConvPoolReLU


class ConvReLU(nn.Sequential):
    def __init__(self, in_size, out_size, kernel_size, stride, padding):
        super(ConvReLU, self).__init__(
            nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU6(inplace=True),
        )


class CNN03(Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.features = Sequential(
            ConvPoolReLU(1, 24, 3, 1, 1, (4, 2), (4, 2)),
            ConvPoolReLU(24, 48, 3, 1, 1, (4, 2), (4, 2)),
            ConvPoolReLU(48, 72, 3, 1, 1, (2, 2), (2, 2)),
            ConvPoolReLU(72, 72, 3, 1, 1, (2, 2), (2, 2)),
            ConvReLU(72, 72, 3, 1, 1),
        )

        self.classifier = Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(720, 10),
        )

    def forward(self, x):
        x = x.view(-1, 1, *x.shape[1:])

        x = self.features(x)
        x = self.classifier(x)

        return x
