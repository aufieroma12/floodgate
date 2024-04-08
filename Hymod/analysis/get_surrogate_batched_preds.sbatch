#!/bin/bash
# get_surrogate_batched_preds.sbatch
#
#SBATCH -J surpreds # Job name for the array
#SBATCH -p janson # Partition to run job on
#SBATCH -c 1 # Number of cores
#SBATCH -t 0-4:30 # Running time of 4.5 hours
#SBATCH --mem 64000 # Memory request of 64 GB
#SBATCH -o Hymod/log/analysis/n_100000_preds/%A_%a.out # Standard output
#SBATCH -e Hymod/log/analysis/n_100000_preds/%A_%a.err # Standard error
# Load modules and run
source .venv/bin/activate && \
python3 Hymod/analysis/get_surrogate_batched_preds.py --train-size=100000 --index=$SLURM_ARRAY_TASK_ID --batch-size=$BATCH_SIZE
