#!/bin/sh

path_py="$HOME/miniconda3/envs/dcase2020/bin/python"

path_script="$HOME/root/task4/standalone/match_multihot_tag.py"
path_dataset="/projets/samova/leocances/dcase2020/DESED/"
path_board="$HOME/root/tensorboard/"

run=$1
experimental=$2
suffix=$3

$path_py $path_script --dataset $path_dataset --logdir $path_board --from_disk False --batch_size_s 64 --batch_size_u 64 --num_workers_s 4 --num_workers_u 4 --nb_epochs 10 --write_results False  --threshold_multihot 0.5 --threshold_confidence 0.5 --run "$run" --experimental "$experimental" --suffix "$suffix"
