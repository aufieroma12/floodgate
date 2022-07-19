export NUM_DATASETS=1
mkdir -p ../log/analysis/n_100000/
sbatch --array=0-999 surrogate_100k.sbatch
