import numpy as np
import joblib
from time import time
import argparse
import os

import warnings
warnings.filterwarnings('ignore')

from src.surrogate import Hymod, KRRcv
from config import Hymod_inputs, KRR_hyperparams, Random_seeds

parser = argparse.ArgumentParser()
parser.add_argument("--n_samples", type=int, required=True, help="Number of training samples.")
parser.add_argument("--index", type=int, required=False, default=-1, help="Index of run.")
args = parser.parse_args()
n = args.n_samples
index = args.index

np.random.seed(Random_seeds["Hymod_train"])

MODEL_DIR = '../models/'
os.makedirs(MODEL_DIR, exist_ok=True)

fstar = Hymod()

xmin = Hymod_inputs["min"]
xmax = Hymod_inputs["max"]
d = xmin.shape[0]
X = np.random.rand(n, d) @ np.diag(xmax - xmin) + np.ones((n, d)) @ np.diag(xmin)
y = fstar.predict(X)

alphas, gammas = KRR_hyperparams[n]

if index >= 0:
    MODEL_DIR = f'{MODEL_DIR}n_{n}/'
    os.makedirs(MODEL_DIR, exist_ok=True)

    alpha = alphas[index // len(gammas)]
    gamma = gammas[index % len(gammas)]

    f = KRRcv([alpha], [gamma])
    t1 = time()
    f.fit(X, y)
    joblib.dump(f, f"{MODEL_DIR}a_{alpha}_g_{gamma}.pkl")

else:
    f = KRRcv(alphas, gammas)
    t1 = time()
    f.fit(X, y)
    joblib.dump(f, f"{MODEL_DIR}n_{n}.pkl") 
    alpha = f.model.best_params_['alpha']
    gamma = f.model.best_params_['gamma']


print(f'n={n}: {time()-t1: .3f} seconds')
print(f'alpha: {alpha}')
print(f'gamma: {gamma}')


