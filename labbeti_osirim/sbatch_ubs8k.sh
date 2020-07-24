#!/bin/sh

run="fm"
suffix="FMV9"

write_results=True
nb_epochs=100

path_torch="/logiciels/containerCollections/CUDA10/pytorch.sif"
path_py="$HOME/miniconda3/envs/dcase2020/bin/python"
path_script="$HOME/root/task4/standalone/main_onehot_tag.py"

path_dataset="/projets/samova/leocances/UrbanSound8K/"
path_board="$HOME/root/tensorboard/UBS8K/fold_10_CNN03/"
path_checkpoint="$HOME/root/task4/models/"

dataset_name="UBS8K"
nb_classes=10
model="CNN03Rot"
num_workers_s=4
num_workers_u=4
checkpoint_metric_name="acc"
batch_size_s=64
batch_size_u=64

scheduler="None"
use_rampup=false
cross_validation=false
threshold_confidence=0.9
nb_rampup_epochs=10
supervised_ratio=0.10
lr=1e-3
shuffle_s_with_u=true
experimental="None"
criterion_name_u="ce"

nb_augms=2
nb_augms_strong=2
lambda_u=1.0
lambda_u1=0.5
lambda_r=0.5


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

srun singularity exec $path_torch $path_py $path_script \
	--run "$run" \
	--suffix "$suffix" \
	--dataset_path $path_dataset \
	--logdir $path_board \
	--checkpoint_path $path_checkpoint \
	--write_results $write_results \
	--nb_epochs $nb_epochs \
	--dataset_name $dataset_name \
	--nb_classes $nb_classes \
	--model $model \
	--num_workers_s $num_workers_s \
	--num_workers_u $num_workers_u \
	--checkpoint_metric_name $checkpoint_metric_name \
	--batch_size_s $batch_size_s \
	--batch_size_u $batch_size_u \
	--scheduler $scheduler \
	--use_rampup $use_rampup \
	--cross_validation $cross_validation \
	--threshold_confidence $threshold_confidence \
	--nb_rampup_epochs $nb_rampup_epochs \
	--supervised_ratio $supervised_ratio \
	--lr $lr \
	--shuffle_s_with_u $shuffle_s_with_u \
	--experimental $experimental \
	--criterion_name_u $criterion_name_u \
	--nb_augms $nb_augms \
	--nb_augms_strong $nb_augms_strong \
	--lambda_u $lambda_u \
	--lambda_u1 $lambda_u1 \
	--lambda_r $lambda_r \

EOT

sbatch $tmp_file
