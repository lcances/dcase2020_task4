#!/bin/sh

path_py="$HOME/miniconda3/envs/dcase2020/bin/python"
path_script="$HOME/root/task4/standalone/augm_search.py"
path_checkpoint="$HOME/root/task4/models/"

$path_py $path_script --checkpoint_path $path_checkpoint --lr 1e-3 --scheduler None
