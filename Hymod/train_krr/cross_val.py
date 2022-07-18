import numpy as np
import joblib
from time import time
import os

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore')

import sys
sys.path.append('../../src/')
sys.path.append('../../config/')

from surrogate import Hymod, KRRcv
from config import Hymod_inputs, KRR_hyperparams


n = 100000
MODEL_DIR = f'../models/n_{n}/'

fstar = Hymod()

xmin = Hymod_inputs["min"]
xmax = Hymod_inputs["max"]
d = xmin.shape[0]
X = np.random.rand(n, d) @ np.diag(xmax - xmin) + np.ones((n, d)) @ np.diag(xmin)
y = fstar.predict(X)

alphas, gammas = KRR_hyperparams[n]
losses = []

for i in range(len(alphas) * len(gammas)):
    alpha = alphas[i // len(gammas)]
    gamma = gammas[i % len(gammas)]
    if os.path.exists(f"{MODEL_DIR}a_{alpha}_g_{gamma}.pkl"):	
        f = joblib.load(f"{MODEL_DIR}a_{alpha}_g_{gamma}.pkl") 
        t1 = time()
        y_preds = f.predict(X)
        mse = np.mean((y - y_preds) ** 2)
        losses.append(mse)
        print(f'alpha={alpha}, gamma={gamma}: {mse} ({time() - t1: .3f} seconds)')
    else:
        losses.append(np.inf)

idx = np.argmin(losses)
alpha = alphas[idx // len(gammas)]
gamma = gammas[idx % len(gammas)]
print(f'Best (alpha, gamma) = ({alpha},{gamma})')
f = joblib.load(f"{MODEL_DIR}a_{alpha}_g_{gamma}.pkl") 
joblib.dump(f, f"../models/n_{n}.pkl") 
