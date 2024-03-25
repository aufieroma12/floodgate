export BATCH_SIZE=1000000
mkdir -p Hymod/log/analysis/n_100000_preds/
sbatch --array=0-99 Hymod/analysis/get_surrogate_batched_preds.sbatch
