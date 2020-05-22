
from dcase2020.pytorch_metrics.metrics import CategoricalAccuracy


class CategoricalConfidenceAccuracy(CategoricalAccuracy):
	""" Just Categorical Accuracy with a binarization with threshold. """

	def __init__(self, confidence: float, epsilon: float = 1e-10):
		super().__init__(epsilon)
		self.confidence = confidence

	def __call__(self, logits, true):
		y_pred = (logits > self.confidence).float()
		y_true = (true > self.confidence).float()
		return super().__call__(y_pred, y_true)
