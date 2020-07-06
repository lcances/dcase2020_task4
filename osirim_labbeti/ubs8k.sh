#!/bin/sh

run=$1
suffix=$2

dataset_name="UBS8K"
nb_classes=10

path_py="$HOME/miniconda3/envs/dcase2020/bin/python"

path_script="$HOME/root/task4/standalone/match_onehot_tag.py"
path_dataset="/projets/samova/leocances/UrbanSound8K/"
path_board="$HOME/root/tensorboard_UBS8K/"

$path_py $path_script --dataset $path_dataset --dataset_name $dataset_name --nb_classes $nb_classes --logdir $path_board --model_name "UBS8KBaseline" --batch_size_s 64 --batch_size_u 64 --num_workers_s 4 --num_workers_u 4 --nb_epochs 10 --write_results False --run "$run" --suffix "$suffix" --use_rampup True
