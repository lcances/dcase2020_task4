#!/bin/sh

run="sf"
suffix="DEBUG"

path_torch="/logiciels/containerCollections/CUDA10/pytorch.sif"
path_py="$HOME/miniconda3/envs/dcase2020/bin/python"
path_script="$HOME/root/task4/standalone/main_onehot_tag.py"

tmp_file=".tmp_sbatch.sh"
name="UT$suffix"
out_file="$HOME/logs/UBS8K_%j_${run}_${suffix}.out"
err_file="$HOME/logs/UBS8K_%j_${run}_${suffix}.err"


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
	--write_results false \
	--debug_mode true \
	--args_filepaths "$HOME/root/task4/labbeti_osirim/args_files/ubs8k.json" "$HOME/root/task4/labbeti_osirim/args_files/ubs8k_${run}.json" \

EOT

sbatch $tmp_file