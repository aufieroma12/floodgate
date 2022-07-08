import numpy as np
import joblib
from time import time
import os

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore')

import sys
sys.path.append('../../')

from src.surrogate import KRRcv
from config.config import KRR_hyperparams


n = 100000
data_path = '../data/outputs/dataset_1001.npy'
model_dir = f'../models/n_{n}/'

alphas, gammas = KRR_hyperparams[n]

data = np.load(data_path)
data = data[:n]
X = data[:,:-1]
y = data[:,-1]

losses = []

for i in range(len(alphas) * len(gammas)):
    alpha = alphas[i // len(gammas)]
    gamma = gammas[i % len(gammas)]
    if os.path.exists(f"{model_dir}a_{alpha}_g_{gamma}.pkl"):	
        f = joblib.load(f"{model_dir}a_{alpha}_g_{gamma}.pkl") 
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
f = joblib.load(f"{model_dir}a_{alpha}_g_{gamma}.pkl") 
joblib.dump(f, f"../models/n_{n}.pkl") 
