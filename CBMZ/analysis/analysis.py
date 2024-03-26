import os
from time import time

import numpy as np

from src.sensitivity import combined_surrogate_methods
from src.surrogate import KelpNN
from config import CBMZ_inputs, Random_seeds

import warnings
warnings.filterwarnings('ignore')


np.random.seed(Random_seeds["CBMZ"])

substances = CBMZ_inputs["labels"]
met_names = CBMZ_inputs["met_labels"]
xmin = CBMZ_inputs["xmin"]
xmax = CBMZ_inputs["xmax"]

n_batches = [4, 7, 40, 79, 391, 625]
batch_size = 128
d = len(substances) # Number of input parameters


DATA_DIR = '../data/outputs/'
OUTPUT_DIR = '../data/analysis/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

n_steps = 3
X = np.load(DATA_DIR + 'conc_inputs.npy')
met = np.load(DATA_DIR + 'met_inputs.npy')[:,:n_steps,:]
y = np.load(DATA_DIR + 'outputs.npy')

f = KelpNN(CBMZ_inputs['MODEL_PATH'])

floodgate_results = []
spf_results = []
panin_results = []

for n in n_batches:
    n *= batch_size
    X_test = X[:n]
    met_test = met[:n]
    y_test = y[:n]

    t1 = time()
    flood, spf, panin = combined_surrogate_methods((X_test, met_test), f, xmin, xmax, Y=y_test, batch_size=batch_size)
    print(f'  n={n}: {time() - t1: .3f} seconds')

    floodgate_results.append(flood)
    spf_results.append(spf)
    panin_results.append(panin)

floodgate_results = np.array(floodgate_results)
spf_results = np.array(spf_results)
panin_results = np.array(panin_results)

np.save(OUTPUT_DIR + 'floodgate.npy', floodgate_results)
np.save(OUTPUT_DIR + 'spf.npy', spf_results)
np.save(OUTPUT_DIR + 'panin.npy', panin_results)
