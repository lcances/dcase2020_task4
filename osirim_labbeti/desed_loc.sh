path_py="$HOME/miniconda3/envs/dcase2020/bin/python"

path_script="$HOME/root/task4/standalone/match_desed_loc.py"
path_dataset="/projets/samova/leocances/dcase2020/DESED/"
path_board="$HOME/root/tensorboard/"
path_checkpoint="$HOME/root/task4/models/"

run=$1
experimental=$2
suffix=$3


$path_py $path_script --dataset $path_dataset --logdir $path_board --debug False --from_disk False --batch_size_s 64 --batch_size_u 64 --num_workers_s 4 --num_workers_u 4 --nb_epochs 10 --path_checkpoint $path_checkpoint --write_results False --use_rampup False --use_alignment False --threshold_multihot 0.5 --threshold_confidence 0.5 --run "$run" --experimental "$experimental" --suffix "$suffix"
