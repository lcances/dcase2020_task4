from easydict import EasyDict as edict


def default_remixmatch_hparams() -> edict:
	# TODO : clean
	hparams = edict()
	hparams.batch_size = 64
	hparams.nb_augms_strong = 8  # In paper : 8
	hparams.sharpen_temp = 0.5
	hparams.mixup_alpha = 0.75
	hparams.lambda_u = 1.0  # In paper : 1.5
	hparams.lambda_u1 = 0.5
	hparams.lambda_r = 0.5
	hparams.history_size = 128

	hparams.lr = 1e-2  # In paper 2e-3
	hparams.weight_decay = 0.0  # In paper 0.02

	# Other param defined for running methods
	hparams.nb_epochs = 100
	hparams.begin_date = ""
	hparams.dataset_name = ""
	hparams.model_name = ""
	hparams.logdir = ""
	return hparams
