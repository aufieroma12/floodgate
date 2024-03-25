export BATCH_SIZE=1000000
mkdir -p ../log/analysis/n_100000_preds/
sbatch --array=0-99 get_surrogate_batched_preds.sbatch
