#!/bin/sh

run=$1
suffix=$2
experimental=$3

path_py="$HOME/miniconda3/envs/dcase2020/bin/python"
path_script="$HOME/root/task4/standalone/match_multihot_tag.py"
path_dataset="/projets/samova/leocances/dcase2020/DESED/"
path_board="$HOME/root/tensorboard_DESED_TAG/"
path_checkpoint="$HOME/root/task4/models/"

$path_py $path_script --dataset $path_dataset --logdir $path_board --path_checkpoint $path_checkpoint --debug False --from_disk False --batch_size_s 64 --batch_size_u 64 --num_workers_s 4 --num_workers_u 4 --nb_epochs 10 --write_results False --use_rampup True --use_sharpen_multihot True --shuffle_s_with_u False --threshold_multihot 0.5 --threshold_confidence 0.9 --mixup_alpha 3.0 --lambda_u 1.0 --run "$run" --experimental "$experimental" --suffix "$suffix"
