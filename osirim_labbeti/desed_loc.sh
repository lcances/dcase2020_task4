path_py="$HOME/miniconda3/envs/dcase2020/bin/python"

path_script="$HOME/root/task4/standalone/match_desed_loc.py"
path_dataset="/projets/samova/leocances/dcase2020/DESED/"
path_board="$HOME/root/tensorboard/"
path_checkpoint="$HOME/root/task4/models/"

run=$1
experimental=$2
suffix=$3

$path_py $path_script --dataset $path_dataset --logdir $path_board --from_disk False --batch_size_s 16 --batch_size_u 112 --num_workers_s 4 --num_workers_u 4 --nb_epochs 100 --path_checkpoint $path_checkpoint --run $run --experimental $experimental --suffix $suffix
