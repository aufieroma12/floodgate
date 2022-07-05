module purge
module load python/3.8.5-fasrc01
module load Anaconda3/2020.11
module load cuda/11.1.0-fasrc01 cudnn/8.0.4.30_cuda11.1-fasrc01
source activate tf_gpu
pip install --upgrade tensorflow-gpu==2.8.0

TF_VERSION=$(conda list tensorflow)
PYTHON_VERSION=$(python --version)
CONDA_VERSION=$(conda --version)
echo 'loaded' ${TF_VERSION:138:31}
echo 'loaded' $PYTHON_VERSION
echo 'loaded' $CONDA_VERSION

