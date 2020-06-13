#!/bin/bash
#SBATCH --job-name=opti_and_eval
#SBATCH --output=opti_and_eval.out
#SBATCH --error=opti_and_eval.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=48CPUNodes

# Sbatch configuration
container=/logiciels/containerCollections/CUDA10/pytorch.sif
python=/users/samova/lcances/.miniconda3/envs/dcase2020/bin/python
script=../standalone/optimization_and_evaluation.py

srun -n1 -N1 singularity exec ${container} ${python} ${script} --model_save ../models/best_dcase2019.torch --model_name dcase2019_model -w 24 -o submission.csv


