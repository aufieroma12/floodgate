import numpy as np
from time import time
import argparse
import os
import joblib

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore')

import sys
sys.path.append('../../')

from src.sensitivity import combined_surrogate_methods
from src.surrogate import KRRcv, Hymod
from config.config import Hymod_inputs


parser = argparse.ArgumentParser()
parser.add_argument("--train_size", type=int, required=True, help="Size of training set.")
parser.add_argument("--num_datasets", type=int, required=True, help="Number of datasets to evaluate.")
parser.add_argument("--index", type=int, required=True, help="Index of run.")
parser.add_argument("--save_data", type=bool, required=False, default=True, help="Whether to save the model inputs and outputs.")
args = parser.parse_args()

index = args.index
num_datasets = args.num_datasets
train_size = args.train_size

start = index * num_datasets
end = (index + 1) * num_datasets

# Input ranges
xmin = Hymod_inputs['min']
xmax = Hymod_inputs['max']
d = xmax.shape[0]

sample_sizes = [100, 250, 500, 1000, 5000, 10000, 50000]
n_max = sample_sizes[-1] # Total number of samples

data_path = '../data/outputs/{}.npy'
model_path = f'../models/n_{train_size}'
output_dir = '../data/analysis/{}/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    os.mkdir(output_dir.format('floodgate'))
    os.mkdir(output_dir.format('surrogate_spf'))
    os.mkdir(output_dir.format('panin'))

fstar = Hymod()
f = joblib.load(model_path)

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
            np.save(data_path.format(i), np.concatenate((X,y.reshape(-1,1)), axis=1))

    print(f'  Train size: {train_size}')
    floodgate_results = []
    spf_results = []
    panin_results = []

    for N in sample_sizes:
        X_test = X[:N]
        y_test = y[:N]
        
        t1 = time()
        flood, spf, panin = combined_surrogate_methods(X_test, f, xmin, xmax, Y=y_test)
        print(f'    N={N}: {time() - t1: .3f} seconds')
        
        floodgate_results.append(flood)
        spf_results.append(spf)
        panin_results.append(panin)
        
    floodgate_results = np.array(floodgate_results)
    spf_results = np.array(spf_results)
    panin_results = np.array(panin_results)

    np.save(output_dir.format('floodgate') + f'{train_size}tr_{i}.npy', floodgate_results)
    np.save(output_dir.format('surrogate_spf') + f'{train_size}tr_{i}.npy', spf_results)
    np.save(output_dir.format('panin') + f'{train_size}tr_{i}.npy', panin_results)

    print('')