import argparse
import os
import sys
from pathlib import Path
from time import time


import joblib
import numpy as np

from config import Hymod_inputs, Random_seeds
from src.surrogate import Surrogate
from src.util import get_knockoffs

# Input ranges
xmin = Hymod_inputs["min"]
xmax = Hymod_inputs["max"]
d = xmax.shape[0]


def get_all_preds(
    X: np.ndarray, f: Surrogate, xmin: np.ndarray, xmax: np.ndarray
) -> np.ndarray:
    n, d = X.shape
    ind = list(np.arange(d))

    print("Making original predictions.")
    all_preds = [f.predict(X)]

    for X_ind in ind:
        print(f"Making knockoff predictions for index {X_ind}.")
        knockoffs = get_knockoffs(X, X_ind, xmin, xmax, 1)
        all_preds.append(f.predict(knockoffs))

    return np.stack(all_preds, axis=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-size",
        type=int,
        required=True,
        help="Size of training set.",
    )
    parser.add_argument(
        "--index",
        type=int,
        required=True,
        help="Index of the batch.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="Number of samples to run in the batch.",
    )
    args = parser.parse_args()
    train_size = args.train_size
    index = args.index
    n = args.batch_size

    OUTPUT_DIR = (
        Path(__file__).parents[1] / "data" / "outputs" / f"surrogate_{train_size}_preds"
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    np.random.seed(Random_seeds["Hymod_gt"] + index)
    print(f"Generating {n} x {d} input matrix")
    X = np.random.rand(n, d) @ np.diag(xmax - xmin) + np.ones((n, d)) @ np.diag(xmin)
    print(f"Input shape: {X.shape}")
    MODEL_PATH = Path(__file__).parents[1] / "models" / f"n_{train_size}.pkl"

    sys.path.append(str(Path(__file__).parents[2] / "src"))
    f = joblib.load(MODEL_PATH)
    print("Loaded model")

    t1 = time()
    preds = get_all_preds(X, f, xmin, xmax)
    print(f"Total model evalutations ({n * (d + 1)}): {(time() - t1): .2f} seconds")

    np.save(OUTPUT_DIR / f"{index}.npy", preds)
