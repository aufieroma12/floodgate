mkdir -p ../log/krr_train/n_100000/
sbatch --array=0-8 train_100k.sbatch
