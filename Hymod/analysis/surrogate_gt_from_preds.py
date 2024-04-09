import os
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parents[1] / "data"
PREDS_DIR = DATA_DIR / "outputs" / "surrogate_100000_preds"
OUTPUT_PATH = DATA_DIR / "analysis" / "ground_truth_surrogate_100000.npy"


def spf_from_preds(all_preds: np.ndarray) -> np.ndarray:
    n = all_preds.shape[0]
    y = all_preds[:, 0]

    V = (n / (n - 1)) * (y - np.mean(y)) ** 2
    V_bar = np.mean(V)

    S_vals = []
    for i in range(1, all_preds.shape[1]):
        Mj = ((y - all_preds[:, i]) ** 2) / 2
        Mj_bar = np.mean(Mj)
        S_vals.append(Mj_bar / V_bar)
    return np.array(S_vals)


if __name__ == "__main__":
    all_preds = []
    for fp in os.listdir(PREDS_DIR):
        all_preds.append(np.load(PREDS_DIR / fp))
    all_preds = np.concatenate(all_preds, axis=0)
    S_vals = spf_from_preds(all_preds)
    np.save(OUTPUT_PATH, S_vals)
