#!/bin/bash
# run.sbatch
#
#SBATCH -J CBMZ # Job name for the array
#SBATCH -p gpu_test # Partition to run job on
#SBATCH -n 1 # Number of nodes
#SBATCH --gres=gpu:1 # Number of GPUs
#SBATCH -t 0-4:00 # Running time of 4 hours
#SBATCH --mem 40000 # Memory request of 40 GB
#SBATCH -o ../log/%A.out # Standard output
#SBATCH -e ../log/%A.err # Standard error
# Make new directory and run
source $BASE_DIR/config/load_modules.sh
python3 analysis.py
