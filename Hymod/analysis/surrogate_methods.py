import numpy as np
from time import time
import argparse
import os
import joblib

import warnings
warnings.filterwarnings('ignore')

from src.sensitivity import combined_surrogate_methods
from src.surrogate import Hymod
from config import Hymod_inputs, Random_seeds


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

DATA_PATH = '../data/outputs/{}.npy'
MODEL_PATH = f'../models/n_{train_size}.pkl'
OUTPUT_DIR = '../data/analysis/{}/{}/'

os.makedirs(OUTPUT_DIR.format('floodgate', train_size), exist_ok=True)
os.makedirs(OUTPUT_DIR.format('spf_surrogate', train_size), exist_ok=True)
os.makedirs(OUTPUT_DIR.format('panin', train_size), exist_ok=True)

fstar = Hymod()
f = joblib.load(MODEL_PATH)

for i in range(start, end):
    print(f'Evaluating dataset {i}:')
    if os.path.exists(DATA_PATH.format(i)):
        data = np.load(DATA_PATH.format(i))
        X = data[:,:-1]
        y = data[:,-1]
        print("  Data read from file.\n")
    else:
        np.random.seed(Random_seeds["Hymod_inputs"] + i)
        X = np.random.rand(n_max, d) @ np.diag(xmax - xmin) + np.ones((n_max, d)) @ np.diag(xmin)
        t1 = time()
        y = fstar.predict(X) 
        print(f"  Total model evalutations ({n_max}): {(time() - t1): .2f} seconds")
        
        if args.save_data:
            np.save(DATA_PATH.format(i), np.concatenate((X,y.reshape(-1,1)), axis=1))

    print(f'  Train size: {train_size}')
    floodgate_results = []
    spf_results = []
    panin_results = []
    np.random.seed(Random_seeds["Hymod_analysis"] + i)

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

    np.save(OUTPUT_DIR.format('floodgate', train_size) + f'{i}.npy', floodgate_results)
    np.save(OUTPUT_DIR.format('spf_surrogate', train_size) + f'{i}.npy', spf_results)
    np.save(OUTPUT_DIR.format('panin', train_size) + f'{i}.npy', panin_results)

    print('')
