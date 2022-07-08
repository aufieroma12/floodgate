#!/bin/bash
# train_100k.sbatch
#
#SBATCH -J krr_100k # Job name for the array
#SBATCH -p shared # Partition to run job on
#SBATCH -c 1 # One core
#SBATCH -t 0-20:00 # Running time of 20 hours
#SBATCH --mem 175000 # Memory request of 175 GB
#SBATCH -o ../log/krr_train/n_100000/%A_%a.out # Standard output
#SBATCH -e ../log/krr_train/n_100000/%A_%a.err # Standard error
# make new directory and run
mkdir -p ../log/krr_train/n_100000/
source ~/load_modules.sh
python3 train.py --n_samples=100000 --index=$SLURM_ARRAY_TASK_ID

