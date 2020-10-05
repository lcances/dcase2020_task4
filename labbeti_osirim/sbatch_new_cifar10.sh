#!/bin/sh

run="sf"
suffix="test_new_wrn_adam_no_sched"

path_torch="/logiciels/containerCollections/CUDA10/pytorch.sif"
path_py="$HOME/miniconda3/envs/dcase2020/bin/python"
path_script="$HOME/root/task4/standalone/main_onehot_tag.py"

tmp_file=".tmp_sbatch.sh"
name="CT$suffix"
out_file="$HOME/logs/CIFAR10_%j_${run}_${suffix}.out"
err_file="$HOME/logs/CIFAR10_%j_${run}_${suffix}.err"


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
	--suffix "$suffix" \
	--checkpoint_path "$HOME/root/task4/models/" \
	--checkpoint_metric_name "acc" \
	--write_results true \
	--debug_mode false \
	--args_filepaths "$HOME/root/task4/labbeti_osirim/args_files/cifar.json" "$HOME/root/task4/labbeti_osirim/args_files/cifar_${run}.json" \

EOT

sbatch $tmp_file
