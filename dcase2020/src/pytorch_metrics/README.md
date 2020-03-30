# pytorch_metrics
Custom implementation of some metrics I use with my PyTorch model

This is a repository for my personnal usage and you can use as it comes.

The functions returns the moving average of the given vaue. To access the raw value of the current mini-batch use the value attribute.

## How to use
```python
from pytorch_metrics.metrics import FScore
...

fscore_func = FScore(epsilon=1e-6)

...

f1 = fscore_func(pred, true)

```

## Metrics available
It contain the following metrics:
- CategoricalAccuracy
- BinaryAccuracy
- Recall
- Precision
- FSCore (F1)
- Ratio (Evaluate the truth number of adversarial generated)

