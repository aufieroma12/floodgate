#!/bin/bash
# cross_val.sbatch
#
#SBATCH -J krr_cv # Job name for the array
#SBATCH -p janson # Partition to run job on
#SBATCH -c 1 # Number of cores
#SBATCH -t 0-2:30 # Running time of 2.5 hours
#SBATCH --mem 175000 # Memory request of 175 GB
#SBATCH -o ../log/krr_train/n_100000/cross_val_%A.out # Standard output
#SBATCH -e ../log/krr_train/n_100000/cross_val_%A.err # Standard error
# Load modules and run
source $BASE_DIR/config/load_modules.sh
python3 cross_val.py

