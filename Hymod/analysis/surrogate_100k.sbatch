#!/bin/bash
# surrogate_100k.sbatch
#
#SBATCH -J sgate10k # Job name for the array
#SBATCH -p shared # Partition to run job on
#SBATCH -c 1 # One core
#SBATCH -t 0-10:00 # Running time of 10 hours
#SBATCH --mem 30000 # Memory request of 30 GB
#SBATCH -o ../log/analysis/n_100000/%A_%a.out # Standard output
#SBATCH -e ../log/analysis/n_100000/%A_%a.err # Standard error
# Make new directory and run!
mkdir -p ../log/analysis/n_100000/
source ~/load_modules.sh .
python3 surrogate_methods.py --train_size=100000 --index=$SLURM_ARRAY_TASK_ID --num_datasets=${NUM_DATASETS} 