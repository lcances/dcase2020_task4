#!/bin/sh

path_torch="/logiciels/containerCollections/CUDA10/pytorch.sif"
path_py="$HOME/miniconda3/envs/dcase2020/bin/python"
path_script="$HOME/root/task4/standalone/augm_search.py"

path_checkpoint="$HOME/root/task4/models/"
dataset_name="UBS8K" # CIFAR10, UBS8K
path_dataset="/projets/samova/leocances/UrbanSound8K/" # CIFAR10, UrbanSound8K
model="CNN03Rot" # WideResNet28Rot, CNN03Rot


$path_py $path_script \
	--checkpoint_path $path_checkpoint \
	--dataset_name $dataset_name \
	--dataset_path $path_dataset \
	--model $model \
	--lr 1e-3 \
	--scheduler None
