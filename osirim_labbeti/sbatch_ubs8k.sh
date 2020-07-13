#!/bin/sh

run=$1
suffix=$2

path_torch="/logiciels/containerCollections/CUDA10/pytorch.sif"
path_py="$HOME/miniconda3/envs/dcase2020/bin/python"

path_script="$HOME/root/task4/standalone/match_onehot_tag.py"
path_dataset="/projets/samova/leocances/UrbanSound8K/"
path_board="$HOME/root/tensorboard_UBS8K/"
path_checkpoint="$HOME/root/task4/models/"

args_file="$HOME/root/task4/osirim_labbeti/args_ubs8k.json"

write_results=True
nb_epochs=100

tmp_file=".tmp_sbatch.sh"
name="UTAG_$run"
out_file="$HOME/logs/UBS8K_%j_$run.out"
err_file="$HOME/logs/UBS8K_%j_$run.err"

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

srun singularity exec $path_torch $path_py $path_script --dataset_path $path_dataset --logdir $path_board --checkpoint_path $path_checkpoint --args_file $args_file --write_results $write_results --nb_epochs $nb_epochs --run "$run" --suffix "$suffix"

EOT

sbatch $tmp_file
