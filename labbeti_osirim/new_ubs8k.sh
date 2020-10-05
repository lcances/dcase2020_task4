#!/bin/sh

run="sf"
suffix="DEBUG"

path_torch="/logiciels/containerCollections/CUDA10/pytorch.sif"
path_py="$HOME/miniconda3/envs/dcase2020/bin/python"
path_script="$HOME/root/task4/standalone/main_onehot_tag.py"


$path_py $path_script \
	--suffix "$suffix" \
	--checkpoint_path "$HOME/root/task4/models/" \
	--checkpoint_metric_name "acc" \
	--write_results false \
	--debug_mode true \
	--args_filepaths "$HOME/root/task4/labbeti_osirim/args_files/ubs8k.json" "$HOME/root/task4/labbeti_osirim/args_files/ubs8k_${run}.json" \
