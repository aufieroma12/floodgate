import argparse
import os
from pathlib import Path
from time import time

import joblib
import numpy as np

from src.sensitivity import combined_surrogate_methods
from src.surrogate import Ishigami

from config import Random_seeds, Ishigami_inputs

parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_size",
    type=int,
    required=True,
    help="Size of training set.",
)
parser.add_argument(
    "--num_datasets",
    type=int,
    required=True,
    help="Number of datasets to evaluate.",
)
parser.add_argument("--index", type=int, required=True, help="Index of run.")
parser.add_argument(
    "--save_data",
    type=bool,
    required=False,
    default=True,
    help="Whether to save the model inputs and outputs.",
)
args = parser.parse_args()

index = args.index
num_datasets = args.num_datasets
train_size = args.train_size

start = index * num_datasets
end = (index + 1) * num_datasets

N = int(1e4)
xmin = Ishigami_inputs["min"]
xmax = Ishigami_inputs["max"]
d = xmax.shape[0]

MODEL_PATH = Path(__file__).parents[1] / "models" / f"n_{train_size}.pkl"
DATA_DIR = Path(__file__).parents[1] / "data"

OUTPUT_DIR = DATA_DIR / "analysis"
FLOODGATE_OUTPUT_DIR = OUTPUT_DIR / "floodgate" / f"{train_size}"
SPF_SURROGATE_OUTPUT_DIR = OUTPUT_DIR / "spf_surrogate" / f"{train_size}"
PANIN_OUTPUT_DIR = OUTPUT_DIR / "panin" / f"{train_size}"

os.makedirs(FLOODGATE_OUTPUT_DIR, exist_ok=True)
os.makedirs(SPF_SURROGATE_OUTPUT_DIR, exist_ok=True)
os.makedirs(PANIN_OUTPUT_DIR, exist_ok=True)

fstar = Ishigami()
f = joblib.load(MODEL_PATH)

for i in range(start, end):
    print(f'Evaluating dataset {i}:')
    data_path = DATA_DIR / f"{i}.npy"
    if os.path.exists(data_path):
        data = np.load(data_path)
        X = data[:, :-1]
        y = data[:, -1]
        print("  Data read from file.\n")
    else:
        np.random.seed(Random_seeds["Ishigami_inputs"] + i)
        X = np.random.rand(N, d) @ np.diag(xmax - xmin) + np.ones((N, d)) @ np.diag(xmin)
        t1 = time()
        y = fstar.predict(X)
        print(f"  Total model evalutations ({N}): {(time() - t1): .2f} seconds")

        if args.save_data:
            np.save(data_path, np.concatenate((X, y.reshape(-1, 1)), axis=1))

    print(f"  Train size: {train_size}")
    floodgate_results = []
    spf_results = []
    panin_results = []
    np.random.seed(Random_seeds["Ishigami_analysis"] + i)

    t1 = time()
    flood, spf, panin = combined_surrogate_methods(X, f, xmin, xmax, Y=y)
    print(f"  Analysis: {time() - t1: .3f} seconds")

    floodgate_results.append(flood)
    spf_results.append(spf)
    panin_results.append(panin)

    floodgate_results = np.array(floodgate_results)
    spf_results = np.array(spf_results)
    panin_results = np.array(panin_results)

    np.save(FLOODGATE_OUTPUT_DIR / f"{i}.npy", floodgate_results)
    np.save(SPF_SURROGATE_OUTPUT_DIR / f"{i}.npy", spf_results)
    np.save(PANIN_OUTPUT_DIR / f"{i}.npy", panin_results)

    print()

