import argparse
import os
from pathlib import Path

import numpy as np

from src.sensitivity import combined_surrogate_methods
from src.surrogate import Ishigami

from config import Random_seeds, Ishigami_inputs

N = int(1e6)
xmin = Ishigami_inputs["min"]
xmax = Ishigami_inputs["max"]
d = xmax.shape[0]

DATA_DIR = Path(__file__).parents[1] / "data_big"
OUTPUT_DIR = DATA_DIR / "analysis"

mu = np.array([0, 0, 0])
Sigma = np.array([[1, 0.3, 0.8], [0.3, 1, 0.5], [0.8, 0.5, 1]])


def _create_input_sample(n):
    return np.random.multivariate_normal(mu, Sigma, n)


def analytical_mse(fstar, f):
    # a_diff = a - 1
    # b_diff = b - 7
    # c_diff = c - 0.1
    # unnormalized = (
    #     4 * (a_diff ** 2) * (np.pi ** 3) +
    #     (4 / 9) * (c_diff ** 2) * (np.pi ** 11) +
    #     (8 / 5) * a_diff * c_diff * (np.pi ** 7) +
    #     3 * (b_diff ** 2) * (np.pi ** 3)
    # )
    # return unnormalized / ((2 * np.pi) ** 3)
    X = _create_input_sample(10000)
    y = fstar.predict(X)
    y_hat = f.predict(X)
    return np.mean((y - y_hat) ** 2)


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

    print(f"noise: {noise}, MSE: {analytical_mse(fstar, f):.5f}")

    for i in range(1000):
        if i % 100 == 0:
            print(f"noise: {noise}, iteration: {i}")

        data_path = DATA_DIR / "outputs" / f"{i}.npy"
        os.makedirs(data_path.parent, exist_ok=True)
        if os.path.exists(data_path):
            data = np.load(data_path)
            X = data[:, :-1]
            y = data[:, -1]
        else:
            np.random.seed(Random_seeds["Ishigami_inputs"] + i)
            X = _create_input_sample(N)
            y = fstar.predict(X)

            if args.save_data:
                np.save(data_path, np.concatenate((X, y.reshape(-1, 1)), axis=1))

        np.random.seed(Random_seeds["Ishigami_analysis"] + i)

        floodgate_results, spf_results, panin_results = combined_surrogate_methods(
            X, f, mu, Sigma, Y=y, K=5
        )

        np.save(FLOODGATE_OUTPUT_DIR / f"{i}.npy", np.array(floodgate_results))
        np.save(SPF_SURROGATE_OUTPUT_DIR / f"{i}.npy", np.array(spf_results))
        np.save(PANIN_OUTPUT_DIR / f"{i}.npy", np.array(panin_results))
