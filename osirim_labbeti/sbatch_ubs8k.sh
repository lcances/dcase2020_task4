#!/bin/sh

run=$1
suffix=$2

dataset_name="UBS8K"
nb_classes=10

path_torch="/logiciels/containerCollections/CUDA10/pytorch.sif"
path_py="$HOME/miniconda3/envs/dcase2020/bin/python"

path_script="$HOME/root/task4/standalone/match_onehot_tag.py"
path_dataset="/projets/samova/leocances/UrbanSound8K/"
path_board="$HOME/root/tensorboard_UBS8K/"
path_checkpoint="$HOME/root/task4/models/"

tmp_file=".tmp_sbatch.sh"
name="TAG_$run"
out_file="$HOME/logs/DESED_TAG_%j.out"
err_file="$HOME/logs/DESED_TAG_%j.err"

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

srun singularity exec $path_torch $path_py $path_script --dataset $path_dataset --dataset_name $dataset_name --nb_classes $nb_classes --logdir $path_board --debug False --model_name "UBS8KBaseline" --batch_size_s 64 --batch_size_u 64 --num_workers_s 4 --num_workers_u 4 --nb_epochs 100 --lambda_u 1.0 --write_results True --run "$run" --suffix "$suffix"

EOT

sbatch $tmp_file
