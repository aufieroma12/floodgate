import numpy as np
from time import time
import argparse
import os

import sys
sys.path.append('../../')

from src.sensitivity import SPF
from src.surrogate import Hymod
from config.config import Hymod_inputs

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
xmin = Hymod_inputs.min
xmax = Hymod_inputs.max
d = xmax.shape[0]

sample_sizes = [100, 250, 500, 1000, 5000, 10000, 50000]
n_max = sample_sizes[-1] # Total number of samples

data_path = '../data/outputs/{}.npy'
output_dir = '../data/analysis/spf/'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

fstar = Hymod()

for i in range(start, end):
    print(f'Evaluating dataset {i}:')
    if os.path.exists(data_path.format(i)):
        data = np.load(data_path.format(i))
        X = data[:,:-1]
        y = data[:,-1]
        print("  Data read from file.\n")
    else:
        X = np.random.rand(n_max, d) @ np.diag(xmax - xmin) + np.ones((n_max, d)) @ np.diag(xmin)
        t1 = time()
        y = fstar.predict(X) 
        print(f"  Total model evalutations ({n_max}): {(time() - t1): .2f} seconds")
        
        if args.save_data:
            np.save(data_path.format(i), np.concatenate((X,y), axis=1))

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
    np.save(output_dir + f'{i}.npy', results)

    print('')
