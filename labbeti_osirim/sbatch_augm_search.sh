#!/bin/sh

path_torch="/logiciels/containerCollections/CUDA10/pytorch.sif"
path_py="$HOME/miniconda3/envs/dcase2020/bin/python"
path_script="$HOME/root/task4/standalone/augm_search.py"

path_checkpoint="$HOME/root/task4/models/"
dataset_name="UBS8K" # CIFAR10, UBS8K
path_dataset="/projets/samova/leocances/UrbanSound8K/" # CIFAR10, UrbanSound8K
model="CNN03Rot" # WideResNet28Rot, CNN03Rot

tmp_file=".tmp_sbatch.sh"
name="AUG_$dataset_name"
out_file="$HOME/logs/AUGM_%j_$dataset_name.out"
err_file="$HOME/logs/AUGM_%j_$dataset_name.err"

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

srun singularity exec $path_torch $path_py $path_script \
	--checkpoint_path $path_checkpoint \
	--dataset_name $dataset_name \
	--dataset_path $path_dataset \
	--model $model \
	--lr 1e-3 \
	--scheduler None

EOT

sbatch $tmp_file
