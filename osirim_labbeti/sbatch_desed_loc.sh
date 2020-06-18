#!/bin/sh

path_torch="/logiciels/containerCollections/CUDA10/pytorch.sif"
path_py="$HOME/miniconda3/envs/dcase2020/bin/python"

path_script="$HOME/root/task4/standalone/match_desed_loc.py"
path_dataset="/projets/samova/leocances/dcase2020/DESED/"
path_board="$HOME/root/tensorboard/"
path_checkpoint="$HOME/root/task4/models/"

run=$1
experimental=$2
suffix=$3

partition="GPUNodes"
tmp_file=".tmp_sbatch.sh"
name="LOC_$run"
out_file="$HOME/logs/DESED_LOC_%j.out"
err_file="$HOME/logs/DESED_LOC_%j.err"

cat << EOT > $tmp_file
#!/bin/sh

#SBATCH --job-name=$name
#SBATCH --output=$out_file
#SBATCH --error=$err_file
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=$partition
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding

# export PYTHONPATH="$HOME/dcase2020_root/:$PYTHONPATH"

module purge
module load singularity/3.0.3
srun singularity exec $path_torch $path_py $path_script --dataset $path_dataset --logdir $path_board --from_disk False --batch_size_s 16 --batch_size_u 112 --num_workers_s 4 --num_workers_u 4 --nb_epochs 100 --path_checkpoint $path_checkpoint --run "$run" --experimental "$experimental" --suffix "$suffix"

EOT

sbatch $tmp_file
