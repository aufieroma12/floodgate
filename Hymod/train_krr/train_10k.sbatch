#!/bin/bash
# train_10k.sbatch
#
#SBATCH -J krr_10k # Job name for the array
#SBATCH -p janson # Partition to run job on
#SBATCH -c 1 # Number of cores
#SBATCH -t 0-2:30 # Running time of 2.5 hours
#SBATCH --mem 20000 # Memory request of 20 GB
#SBATCH -o ../log/krr_train/n_10000/%A.out # Standard output
#SBATCH -e ../log/krr_train/n_10000/%A.err # Standard error
# Load modules and run
source $BASE_DIR/config/load_modules.sh
python3 train.py --n_samples=10000

