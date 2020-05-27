
from time import time
from torch.nn import Module
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from typing import Callable

from dcase2020.pytorch_metrics.metrics import Metrics


def val(model: Module, acti_fn: Callable, loader: DataLoader, nb_classes: int, metrics: Metrics, epoch: int) -> (list, list):
	metrics.reset()
	val_start = time()
	model.eval()

	accuracies, maxs = [], []
	iter_val = iter(loader)
	for i, (x, y) in enumerate(iter_val):
		x, y = x.cuda().float(), y.cuda().long()
		y = one_hot(y, nb_classes)

		logits_x = model(x)
		pred_x = acti_fn(logits_x)
		accuracy_val = metrics(pred_x, y)

		accuracies.append(metrics.value.item())
		maxs.append(pred_x.max(dim=1)[0].mean().item())
		print("Epoch {}, {:d}% \t val_acc: {:.4e} - took {:.2f}s".format(
			epoch + 1,
			int(100 * (i + 1) / len(loader)),
			accuracy_val,
			time() - val_start,
		), end="\r")

	print("")
	return accuracies, maxs
