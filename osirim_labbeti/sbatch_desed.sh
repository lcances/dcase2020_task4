#!/bin/sh

path_torch="/logiciels/containerCollections/CUDA10/pytorch.sif"
path_py="$HOME/miniconda3/envs/dcase2020/bin/python"

path_script="$HOME/root/task4/standalone/match_desed.py"
path_dataset="/projets/samova/leocances/dcase2020/DESED/"
path_board="$HOME/root/tensorboard/"

run=$1
experimental=$2
suffix=$3

partition="GPUNodes"
tmp_file=".tmp_sbatch.sh"

cat << EOT > $tmp_file
#!/bin/sh

#SBATCH --job-name=DESED_$run
#SBATCH --output=logs/DESED_%j.out
#SBATCH --error=logs/DESED_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=$partition
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding

# export PYTHONPATH="$HOME/dcase2020_root/:$PYTHONPATH"

module purge
module load singularity/3.0.3

srun singularity exec $path_torch $path_py $path_script --dataset $path_dataset --logdir $path_board --from_disk False --batch_size_s 64 --batch_size_u 64 --num_workers_s 4 --num_workers_u 4 --nb_epochs 100 --run $run --experimental $experimental --suffix $suffix

EOT

sbatch $tmp_file
