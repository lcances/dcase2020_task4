#!/bin/sh

path_py="$HOME/miniconda3/envs/dcase2020/bin/python"

path_script="$HOME/root/task4/standalone/match_onehot_tag.py"
path_dataset="/projets/samova/leocances/CIFAR10/"
path_board="$HOME/root/tensorboard_CIFAR10/"

dataset_name="CIFAR10"
nb_classes=10

run=$1
suffix=$2

$path_py $path_script --dataset $path_dataset --dataset_name $dataset_name --nb_classes $nb_classes --logdir $path_board --debug False --model_name "VGG11" --batch_size_s 64 --batch_size_u 64 --num_workers_s 4 --num_workers_u 4 --nb_epochs 10 --write_results False --use_rampup True --cross_validation False --run "$run" --suffix "$suffix"
