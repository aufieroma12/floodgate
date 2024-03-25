import numpy as np
from time import time
import os

from src.sensitivity import SPF
from src.surrogate import Hymod
from config import Hymod_inputs


# Input ranges
xmin = Hymod_inputs['min']
xmax = Hymod_inputs['max']
d = xmax.shape[0]

n = int(1e8) # Number of samples to use 
np.random.seed(n)

OUTPUT_DIR = '../data/analysis/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

fstar = Hymod()
X = np.random.rand(n, d) @ np.diag(xmax - xmin) + np.ones((n, d)) @ np.diag(xmin)

t1 = time()
results = SPF(X, fstar, xmin, xmax, alpha=1)
print(f"Total model evalutations ({n * (d + 1)}): {(time() - t1): .2f} seconds")

results = np.array(results)[:,0]
np.save(OUTPUT_DIR + 'ground_truth.npy', results)

