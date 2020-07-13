from easydict import EasyDict as edict


def default_fixmatch_hparams() -> edict:
	# TODO : clean
	hparams = edict()
	hparams.batch_size_s = 64
	hparams.batch_size_u = 64
	hparams.lambda_u = 1.0
	hparams.threshold_confidence = 0.95  # tau
	hparams.mode = "onehot"
	hparams.threshold_multihot = 0.5

	hparams.lr = 1e-3  # learning rate, eta
	hparams.weight_decay = 0.0  # 1e-4
	hparams.momentum = 0  # in paper : SGD hyperparameter beta = 0.9

	# Other param defined for running methods
	hparams.train_name = "FixMatch"
	hparams.nb_epochs = 100
	hparams.begin_date = ""
	hparams.dataset_name = ""
	hparams.model = ""
	hparams.logdir = ""
	return hparams
