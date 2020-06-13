#!/bin/bash

#SBATCH --job-name=demo_osirim
#SBATCH --output=demo_osirim.out
#SBATCH --error=demo_osirim.err

#SBATCH --partition=64CPUNodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20


conda activate desed

cd ../dataset_desed
srun bash create_dcase2020_dataset.sh
