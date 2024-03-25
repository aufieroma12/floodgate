import argparse
import os
import sys
from pathlib import Path
from time import time


import joblib
import numpy as np

from src.sensitivity import SPF
from config import Hymod_inputs


# Input ranges
xmin = Hymod_inputs["min"]
xmax = Hymod_inputs["max"]
d = xmax.shape[0]

n = int(1e8) # Number of samples to use

OUTPUT_DIR = Path(__file__).parents[1] / "data" / "analysis"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-size",
        type=int,
        required=True,
        help="Size of training set.",
    )
    args = parser.parse_args()
    train_size = args.train_size

    np.random.seed(n)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    MODEL_PATH = Path(__file__).parents[1] / "models" / f"n_{train_size}.pkl"

    sys.path.append(str(Path(__file__).parents[2] / "src"))
    f = joblib.load(MODEL_PATH)
    X = np.random.rand(n, d) @ np.diag(xmax - xmin) + np.ones((n, d)) @ np.diag(xmin)

    t1 = time()
    results = SPF(X, f, xmin, xmax, alpha=1)
    print(f"Total model evalutations ({n * (d + 1)}): {(time() - t1): .2f} seconds")

    results = np.array(results)[:,0]
    np.save(OUTPUT_DIR / f"ground_truth_surrogate_{train_size}.npy", results)

