import numpy as np
from time import time
import os

import sys
sys.path.append('../../src')
sys.path.append('../../config')

from sensitivity import SPF
from surrogate import Hymod
from config import Hymod_inputs


# Input ranges
xmin = Hymod_inputs['min']
xmax = Hymod_inputs['max']
d = xmax.shape[0]

n = int(1e7)

output_dir = '../data/analysis/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

fstar = Hymod()
X = np.random.rand(n, d) @ np.diag(xmax - xmin) + np.ones((n, d)) @ np.diag(xmin)

t1 = time()
results = SPF(X, fstar, xmin, xmax, alpha=1)
print(f"Total model evalutations ({n * (d + 1)}): {(time() - t1): .2f} seconds")

np.save(output_dir + 'ground_truth.npy', results)

