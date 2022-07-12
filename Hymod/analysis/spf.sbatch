#!/bin/bash
# spf.sbatch
#
#SBATCH -J spf # Job name for the array
#SBATCH -p shared # Partition to run job on
#SBATCH -c 1 # Number of cores
#SBATCH -t 0-4:00 # Running time of 4 hours
#SBATCH --mem 10000 # Memory request of 10 GB
#SBATCH -o ../log/analysis/SPF/%A_%a.out # Standard output
#SBATCH -e ../log/analysis/SPF/%A_%a.err # Standard error
# Make new directory and run
mkdir -p ../log/analysis/SPF/
source $BASE_DIR/load_modules.sh
python3 non_surrogate.py --index=$SLURM_ARRAY_TASK_ID --num_datasets=$NUM_DATASETS
