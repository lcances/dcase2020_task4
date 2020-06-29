#!/bin/sh

#SBATCH --job-name=LOC_fm
#SBATCH --output=/users/samova/elabbe/logs/DESED_LOC_%j.out
#SBATCH --error=/users/samova/elabbe/logs/DESED_LOC_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
# For GPU nodes
#SBATCH --partition="GPUNodes"
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding

module purge
module load singularity/3.0.3

srun singularity exec /logiciels/containerCollections/CUDA10/pytorch.sif /users/samova/elabbe/miniconda3/envs/dcase2020/bin/python /users/samova/elabbe/root/task4/standalone/match_desed_loc.py --dataset /projets/samova/leocances/dcase2020/DESED/ --logdir /users/samova/elabbe/root/tensorboard/ --debug False --from_disk False --batch_size_s 64 --batch_size_u 64 --num_workers_s 4 --num_workers_u 4 --nb_epochs 100 --path_checkpoint /users/samova/elabbe/root/task4/models/ --write_results True --use_rampup True --use_alignment False --threshold_multihot 0.5 --threshold_confidence 0.999 --run "fm" --experimental "V3" --suffix "very_high_thres_with_RampUp_V3"

