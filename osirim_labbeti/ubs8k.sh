#!/bin/sh

run=$1
suffix=$2

path_torch="/logiciels/containerCollections/CUDA10/pytorch.sif"
path_py="$HOME/miniconda3/envs/dcase2020/bin/python"

path_script="$HOME/root/task4/standalone/match_onehot_tag.py"
path_dataset="/projets/samova/leocances/UrbanSound8K/"
path_board="$HOME/root/tensorboard_UBS8K/"
path_checkpoint="$HOME/root/task4/models/"

args_file="$HOME/root/task4/osirim_labbeti/args_ubs8k.json"

write_results=False
nb_epochs=20

$path_py $path_script --dataset_path $path_dataset --logdir $path_board --args_file $args_file --write_results $write_results --nb_epochs $nb_epochs --run "$run" --suffix "$suffix"
