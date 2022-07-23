#!/bin/bash
# ground_truth.sbatch
#
#SBATCH -J gt_hymod # Job name for the array
#SBATCH -p shared # Partition to run job on
#SBATCH -c 1 # Number of cores
#SBATCH -t 0-20:00 # Running time of 20 hours
#SBATCH --mem 50000 # Memory request of 50 GB
#SBATCH -o ../log/analysis/ground_truth/%A.out # Standard output
#SBATCH -e ../log/analysis/ground_truth/%A.err # Standard error
# Load modules and run
source $BASE_DIR/config/load_modules.sh
python3 ground_truth.py
