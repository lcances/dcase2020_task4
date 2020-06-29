#!/bin/sh

path_torch="/logiciels/containerCollections/CUDA10/pytorch.sif"
path_py="$HOME/miniconda3/envs/dcase2020/bin/python"

path_script="$HOME/root/task4/standalone/match_onehot_tag.py"
path_dataset="/projets/samova/leocances/CIFAR10/"
path_board="$HOME/root/tensorboard/"

tmp_file=".tmp_sbatch.sh"
name="CIFAR10"
out_file="$HOME/logs/CIFAR10_%j.out"
err_file="$HOME/logs/CIFAR10_%j.err"

cat << EOT > $tmp_file
#!/bin/sh

#SBATCH --job-name=$name
#SBATCH --output=$out_file
#SBATCH --error=$err_file
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
# For GPU nodes
#SBATCH --partition="GPUNodes"
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding

module purge
module load singularity/3.0.3
srun singularity exec $path_torch $path_py $path_script --dataset $path_dataset --logdir $path_board --model_name "VGG11" --batch_size_s 64 --batch_size_u 64 --num_workers_s 4 --num_workers_u 4 --nb_epochs 100  

EOT

sbatch $tmp_file
