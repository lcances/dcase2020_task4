from easydict import EasyDict as edict


def default_fixmatch_hparams() -> edict:
	hparams = edict()
	hparams.batch_size = 64
	hparams.lambda_u = 1.0
	hparams.momentum = 0  # in paper : beta = 0.9
	hparams.threshold_mask = 0.95  # tau
	hparams.lr = 0.03  # learning rate, eta
	hparams.weight_decay = 1e-4
	hparams.mode = "onehot"
	hparams.threshold_multihot = 0.5

	# Other param defined for running methods
	hparams.nb_epochs = 100
	hparams.begin_date = ""
	hparams.dataset_name = ""
	hparams.model_name = ""
	hparams.logdir = ""
	return hparams
