import numpy as np
import joblib
from time import time
import argparse

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore')

import sys
sys.path.append('../../')

from src.surrogate import Hymod, KRRcv
from config.config import Hymod_inputs, KRR_hyperparams


parser = argparse.ArgumentParser()
parser.add_argument("--n_samples", type=int, required=True, help="Number of training samples.")
parser.add_argument("--index", type=int, required=False, default=-1, help="Index of run.")
args = parser.parse_args()
n = args.n_samples
index = args.index

MODEL_DIR = '../models/'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

fstar = Hymod()

xmin = Hymod_inputs["min"]
xmax = Hymod_inputs["max"]
d = xmin.shape[0]
X = np.random.rand(n, d) @ np.diag(xmax - xmin) + np.ones((n, d)) @ np.diag(xmin)
y = fstar.predict(X)

alphas, gammas = KRR_hyperparams[n]

if index >= 0:
    model_dir = f'{model_dir}n_{n}/'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    alpha = alphas[index // len(gammas)]
    gamma = gammas[index % len(gammas)]

    f = KRRcv([alpha], [gamma])
    t1 = time()
    f.fit(X, y)
    joblib.dump(f, f"{model_dir}a_{alpha}_g_{gamma}.pkl")

else:
    f = KRRcv(alphas, gammas)
    t1 = time()
    f.fit(X, y)
    joblib.dump(f, f"{model_dir}n_{n}.pkl") 
    alpha = f.model.best_params_['alpha']
    gamma = f.model.best_params_['gamma']


print(f'n={n}: {time()-t1: .3f} seconds')
print(f'alpha: {alpha}')
print(f'gamma: {gamma}')


