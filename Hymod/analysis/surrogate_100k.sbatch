#!/bin/bash
# surrogate_100k.sbatch
#
#SBATCH -J sgate100k # Job name for the array
#SBATCH -p janson # Partition to run job on
#SBATCH -c 1 # Number of cores
#SBATCH -t 0-20:00 # Running time of 20 hours
#SBATCH --mem 100000 # Memory request of 100 GB
#SBATCH -o ../log/analysis/n_100000/%A_%a.out # Standard output
#SBATCH -e ../log/analysis/n_100000/%A_%a.err # Standard error
# Load modules and run
source $BASE_DIR/config/load_modules.sh
python3 surrogate_methods.py --train_size=100000 --index=$SLURM_ARRAY_TASK_ID --num_datasets=$NUM_DATASETS 
