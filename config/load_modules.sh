module purge
module load python/3.8.5-fasrc01
module load Anaconda3/2020.11
module load cuda/11.1.0-fasrc01 cudnn/8.0.4.30_cuda11.1-fasrc01
pip install --upgrade tensorflow-gpu==2.8.0

TF_VERSION=$(pip list | grep tensorflow-gpu)
PYTHON_VERSION=$(python --version)
CONDA_VERSION=$(conda --version)
echo 'loaded' $PYTHON_VERSION
echo 'loaded' $CONDA_VERSION
echo 'loaded' ${TF_VERSION:0:40}