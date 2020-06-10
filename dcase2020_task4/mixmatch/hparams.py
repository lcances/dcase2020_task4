from easydict import EasyDict as edict


def default_mixmatch_hparams() -> edict:
	# TODO : clean
	hparams = edict()
	hparams.batch_size = 64
	hparams.nb_augms = 2
	hparams.sharpen_temp = 0.5
	hparams.mixup_alpha = 0.75
	hparams.lambda_u_max = 10.0  # In paper : 75
	hparams.criterion_name_u = "sqdiff"  # In paper : sqdiff, can use crossentropy

	hparams.lr = 1e-2
	hparams.weight_decay = 0.0  # In paper : 8e-4

	# Other param defined for running methods
	hparams.nb_epochs = 100
	hparams.begin_date = ""
	hparams.dataset_name = ""
	hparams.model_name = ""
	hparams.logdir = ""
	return hparams
