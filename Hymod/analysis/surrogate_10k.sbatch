#!/bin/bash
# surrogate_10k.sbatch
#
#SBATCH -J sgate10k # Job name for the array
#SBATCH -p shared # Partition to run job on
#SBATCH -c 1 # One core
#SBATCH -t 0-5:00 # Running time of 5 hours
#SBATCH --mem 25000 # Memory request of 25 GB
#SBATCH -o ../log/analysis/n_10000/%A_%a.out # Standard output
#SBATCH -e ../log/analysis/n_10000/%A_%a.err # Standard error
# Make new directory and run
mkdir -p ../log/analysis/n_10000/
source ~/load_modules.sh .
python3 surrogate_methods.py --train_size=10000 --index=$SLURM_ARRAY_TASK_ID --num_datasets=${NUM_DATASETS} 
