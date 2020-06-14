#!/bin/bash

function show_help {
    echo "usage:  $BASH_SOURCE -n <node_name> -N <ntask> -p <parser_ratio> -m <model_name>"
    echo "    -s SAVE path to the model saved .torch"
    echo "    -m MODEL name of the class or function describing the model"
    echo "    -w WORKERS number of worker to use for the optimization"
    echo "    -o OUTPUT name of the file where the submission will be written"
    echo "    -p partition where the job will be executed. must be a CPU partition"
}

# default parameters
OUTPUT="submission.csv"
SAVE="../models/best_dcase2019.torch"
MODEL="dcase2019_model"
WORKERS=12
PARTITION="48CPUNodes"

while getopts ":s:m:w:o:p:" arg; do
  case $arg in
    s) SAVE=$OPTARG;;
    m) MODEL=$OPTARG;;
    w) WORKERS=$OPTARG;;
    o) OUTPUT=$OPTARG;;
    p) PARTITION=$OPTARG;;
    *) 
        echo "invalide option" 1>&2
        show_help
        exit 1
        ;;
  esac
done

cat << EOT > .sbatch_tmp.sh
#!/bin/bash
#SBATCH --job-name=opti_and_eval
#SBATCH --output=opti_and_eval.out
#SBATCH --error=opti_and_eval.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$WORKERS
#SBATCH --partition=$PARTITION

# Sbatch configuration
container=/logiciels/containerCollections/CUDA10/pytorch.sif
python=/users/samova/lcances/.miniconda3/envs/dcase2020/bin/python
script=../standalone/optimization_and_evaluation.py

srun -n1 -N1 singularity exec \${container} \${python} \${script} --model_save ${SAVE} --model_name ${MODEL} -w ${WORKERS} -o ${OUTPUT}


EOT

echo "sbatch store in .sbatch_tmp.sh"
sbatch .sbatch_tmp.sh
