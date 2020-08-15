#!/bin/sh

run="mm"
suffix="DEBUG"

path_torch="/logiciels/containerCollections/CUDA10/pytorch.sif"
path_py="$HOME/miniconda3/envs/dcase2020/bin/python"
path_script="$HOME/root/task4/standalone/main_onehot_tag.py"


$path_py $path_script \
	--run "$run" \
	--suffix "$suffix" \
	--nb_epochs 10 \
	--experimental "None" \
	--optimizer "Adam" \
	--scheduler "None" \
	--use_rampup false \
	--nb_rampup_steps 10 \
	--cross_validation false \
	--threshold_confidence 0.9 \
	--lr 1e-3 \
	--nb_augms 2 \
	--nb_augms_strong 8 \
	--lambda_u 1.0 \
	--lambda_u1 0.5 \
	--lambda_r 0.5 \
	--batch_size_s 64 \
	--batch_size_u 64 \
	--rampup_each_epoch true \
	--shuffle_s_with_u true \
	--criterion_name_u "ce" \
	--dataset_path "/projets/samova/leocances/UrbanSound8K/" \
	--logdir "$HOME/root/tensorboard/UBS8K/fold_10_CNN03/" \
	--checkpoint_path "$HOME/root/task4/models/" \
	--dataset_name "UBS8K" \
	--nb_classes 10 \
	--supervised_ratio 0.10 \
	--model "CNN03Rot" \
	--num_workers_s 4 \
	--num_workers_u 4 \
	--checkpoint_metric_name "acc" \
	--write_results false \
	--debug_mode true \
