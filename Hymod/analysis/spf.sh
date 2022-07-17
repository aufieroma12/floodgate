export NUM_DATASETS=200
mkdir -p ../log/analysis/SPF/
sbatch --array=0-4 spf.sbatch
