import argparse
import os
from pathlib import Path

import numpy as np

from src.analytical import a, b, c, alpha, beta, gamma, analytical_mse
from src.distribution import MultivariateNormal
from src.sensitivity import combined_surrogate_methods
from src.surrogate import Surrogate
from config import Random_seeds

N = int(1e6)

DATA_DIR = Path(__file__).parents[1] / "data"
OUTPUT_DIR = DATA_DIR / "analysis"

mu = np.array([0, 0, 0])
Sigma = np.array([[1, a, b], [a, 1, c], [b, c, 1]])


class PolynomialFunction(Surrogate):

    def __init__(self, alpha=alpha, beta=beta, gamma=gamma):
        super().__init__(None)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def fit(self, X, y):
        pass

    def predict(self, X):
        return (
            self.alpha * X[:, 0] +
            self.beta * (X[:, 1]**2) +
            self.gamma * (X[:, 2]**4) * X[:, 0]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--noise",
        type=float,
        required=True,
        help="Noise to add to the polynomial coefficients for the surrogate.",
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

    mvn = MultivariateNormal(mu, Sigma)
    fstar = PolynomialFunction()
    f = PolynomialFunction(alpha + noise, beta + 2 * noise, gamma - 0.5 * noise)

    print(f"noise: {noise}, MSE: {analytical_mse(f.alpha, f.beta, f.gamma):.5f}")

    for i in range(1000):
        data_path = DATA_DIR / "outputs" / f"{i}.npy"
        os.makedirs(data_path.parent, exist_ok=True)
        if os.path.exists(data_path):
            data = np.load(data_path)
            X = data[:, :-1]
            y = data[:, -1]
        else:
            np.random.seed(Random_seeds["analytical_polynomial_inputs"] + i)
            X = mvn.joint_sample(N)
            y = fstar.predict(X)

            if args.save_data:
                np.save(data_path, np.concatenate((X, y.reshape(-1, 1)), axis=1))

        np.random.seed(Random_seeds["analytical_polynomial_analysis"] + i)

        floodgate_results, spf_results, panin_results = combined_surrogate_methods(
            X, f, mvn, Y=y, K=50
        )

        np.save(FLOODGATE_OUTPUT_DIR / f"{i}.npy", np.array(floodgate_results))
        np.save(SPF_SURROGATE_OUTPUT_DIR / f"{i}.npy", np.array(spf_results))
        np.save(PANIN_OUTPUT_DIR / f"{i}.npy", np.array(panin_results))
