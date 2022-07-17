export NUM_DATASETS=2
mkdir -p ../log/analysis/n_10000/
sbatch --array=0-499 surrogate_10k.sbatch
