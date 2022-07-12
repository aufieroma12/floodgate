import numpy as np
from time import time
import argparse
import os

import sys
sys.path.append('../../src')
sys.path.append('../../config')

from sensitivity import SPF
from surrogate import Hymod
from config import Hymod_inputs

parser = argparse.ArgumentParser()
parser.add_argument("--num_datasets", type=int, required=True, help="Number of datasets to evaluate.")
parser.add_argument("--index", type=int, required=True, help="Index of run.")
parser.add_argument("--save_data", type=bool, required=False, default=True, help="Whether to save the model inputs and outputs.")
args = parser.parse_args()

index = args.index
num_datasets = args.num_datasets

start = index * num_datasets
end = (index + 1) * num_datasets

# Input ranges
xmin = Hymod_inputs['min']
xmax = Hymod_inputs['max']
d = xmax.shape[0]

sample_sizes = [100, 250, 500, 1000, 5000, 10000, 50000]
n_max = sample_sizes[-1] # Total number of samples

DATA_PATH = '../data/outputs/{}.npy'
OUTPUT_DIR = '../data/analysis/spf/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

fstar = Hymod(obsPath="../data/inputs/LeafCatch.txt")

for i in range(start, end):
    print(f'Evaluating dataset {i}:')
    if os.path.exists(DATA_PATH.format(i)):
        data = np.load(DATA_PATH.format(i))
        X = data[:,:-1]
        y = data[:,-1]
        print("  Data read from file.\n")
    else:
        X = np.random.rand(n_max, d) @ np.diag(xmax - xmin) + np.ones((n_max, d)) @ np.diag(xmin)
        t1 = time()
        y = fstar.predict(X) 
        print(f"  Total model evalutations ({n_max}): {(time() - t1): .2f} seconds")
        
        if args.save_data:
            np.save(DATA_PATH.format(i), np.concatenate((X,y.reshape(-1,1)), axis=1))

    results = []
    print("  Analysis:")

    for N in sample_sizes:
        n = N // (d + 1)
        X_test = X[:n]
        y_test = y[:n]
        
        t1 = time()
        results.append(SPF(X_test, fstar, xmin, xmax, Y=y_test))
        print(f'    N={N}: {time() - t1: .3f} seconds')
        
    results = np.array(results)
    np.save(OUTPUT_DIR + f'{i}.npy', results)

    print('')
