import argparse
import os
from pathlib import Path
from time import time

import numpy as np

from src.sensitivity import combined_surrogate_methods
from src.surrogate import Ishigami

from config import Random_seeds, Ishigami_inputs

N = int(1e4)
xmin = Ishigami_inputs["min"]
xmax = Ishigami_inputs["max"]
d = xmax.shape[0]

DATA_DIR = Path(__file__).parents[1] / "data"
OUTPUT_DIR = DATA_DIR / "analysis"


def analytical_mse(a, b, c):
    a_diff = a - 1
    b_diff = b - 7
    c_diff = c - 0.1
    unnormalized = (
        4 * (a_diff ** 2) * (np.pi ** 3) +
        (4 / 9) * (c_diff ** 2) * (np.pi ** 11) +
        (8 / 5) * a_diff * c_diff * (np.pi ** 7) +
        3 * (b_diff ** 2) * (np.pi ** 3)
    )
    return unnormalized / ((2 * np.pi) ** 3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--noise",
        type=float,
        required=True,
        help="Noise to add to the Ishigami coefficients for the surrogate.",
    )
    parser.add_argument(
        "--save-data",
        action="store_true",
        default=False,
        help="Whether to save the model inputs and outputs.",
    )
    args = parser.parse_args()
    noise = args.noise

    FLOODGATE_OUTPUT_DIR = OUTPUT_DIR / "floodgate" / f"{noise}"
    SPF_SURROGATE_OUTPUT_DIR = OUTPUT_DIR / "spf_surrogate" / f"{noise}"
    PANIN_OUTPUT_DIR = OUTPUT_DIR / "panin" / f"{noise}"

    os.makedirs(FLOODGATE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(SPF_SURROGATE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(PANIN_OUTPUT_DIR, exist_ok=True)

    fstar = Ishigami()
    f = Ishigami(1 + noise, 7 + 2 * noise, 0.1 - 0.5 * noise)

    print(f"MSE: {analytical_mse(f.a, f.b, f.c):.5f}")

    for i in range(1000):
        data_path = DATA_DIR / "outputs" / f"{i}.npy"
        os.makedirs(data_path.parent, exist_ok=True)
        if os.path.exists(data_path):
            data = np.load(data_path)
            X = data[:, :-1]
            y = data[:, -1]
        else:
            np.random.seed(Random_seeds["Ishigami_inputs"] + i)
            X = np.random.rand(N, d) @ np.diag(xmax - xmin) + np.ones((N, d)) @ np.diag(xmin)
            t1 = time()
            y = fstar.predict(X)

            if args.save_data:
                np.save(data_path, np.concatenate((X, y.reshape(-1, 1)), axis=1))

        np.random.seed(Random_seeds["Ishigami_analysis"] + i)

        t1 = time()
        floodgate_results, spf_results, panin_results = combined_surrogate_methods(
            X, f, xmin, xmax, Y=y, K=1
        )

        np.save(FLOODGATE_OUTPUT_DIR / f"{i}.npy", np.array(floodgate_results))
        np.save(SPF_SURROGATE_OUTPUT_DIR / f"{i}.npy", np.array(spf_results))
        np.save(PANIN_OUTPUT_DIR / f"{i}.npy", np.array(panin_results))
