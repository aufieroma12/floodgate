#!/bin/bash
# get_surrogate_batched_preds.sbatch
#
#SBATCH -J surpreds # Job name for the array
#SBATCH -p janson # Partition to run job on
#SBATCH -c 1 # Number of cores
#SBATCH -t 0-3:00 # Running time of 3 hours
#SBATCH --mem 2000 # Memory request of 2 GB
#SBATCH -o ../log/analysis/n_100000_preds/%A_%a.out # Standard output
#SBATCH -e ../log/analysis/n_100000_preds/%A_%a.err # Standard error
# Load modules and run
source $BASE_DIR/config/load_modules.sh
python3 get_surrogate_batched_preds.py --train-size=100000 --index=$SLURM_ARRAY_TASK_ID --batch-size=$BATCH_SIZE
