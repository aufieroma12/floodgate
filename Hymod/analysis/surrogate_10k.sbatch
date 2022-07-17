#!/bin/bash
# surrogate_10k.sbatch
#
#SBATCH -J sgate10k # Job name for the array
#SBATCH -p janson # Partition to run job on
#SBATCH -c 1 # Number of cores
#SBATCH -t 0-4:00 # Running time of 4 hours
#SBATCH --mem 25000 # Memory request of 25 GB
#SBATCH -o ../log/analysis/n_10000/%A_%a.out # Standard output
#SBATCH -e ../log/analysis/n_10000/%A_%a.err # Standard error
# Make new directory and run
source $BASE_DIR/config/load_modules.sh
python3 surrogate_methods.py --train_size=10000 --index=$SLURM_ARRAY_TASK_ID --num_datasets=$NUM_DATASETS 
