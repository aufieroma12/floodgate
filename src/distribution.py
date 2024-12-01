"""Classes to sample from various joint distributions."""
from abc import ABC, abstractmethod
from typing import Union

import numpy as np


class Distribution(ABC):

    @abstractmethod
    def joint_sample(self, n: int) -> np.ndarray:
        """Create a sample of size n from the joint distribution."""

    @abstractmethod
    def conditional_sample(self, inputs: np.ndarray, idx: int) -> np.ndarray:
        """Create a sample of the input at index idx conditioned on the other inputs."""

    def get_knockoffs(
        self, inputs: Union[tuple, np.ndarray], idx: int, k: int
    ) -> Union[tuple, np.ndarray]:
        """Generate k knockoff samples for each row of the inputs by resampling input at idx."""
        if isinstance(inputs, tuple):
            X, meta = inputs
            meta_new = np.zeros((meta.shape[0] * k, meta.shape[1], meta.shape[2]))
        else:
            X = inputs

        n, d = X.shape
        ind = list(np.arange(d))
        Z_ind = ind[:idx] + ind[(idx + 1):]
        
        knockoffs = np.zeros(((n * k), d))
        for i in range(n):
            knockoffs[(i * k):((i + 1) * k), Z_ind] = inputs[i, Z_ind]
        knockoffs[:, idx] = self.conditional_sample(knockoffs, idx)

        if isinstance(inputs, tuple):
            for i in range(n):
                meta_new[(i * k):((i + 1) * k), :] = meta[i, :, :]
            knockoffs = (knockoffs, meta_new)

        return knockoffs


class IndependentUniform(Distribution):

    def __init__(self, xmin: np.ndarray, xmax: np.ndarray) -> None:
        self.xmin = xmin
        self.xmax = xmax
        
    def conditional_sample(self, inputs: np.ndarray, idx: int) -> np.ndarray:
        return np.random.rand(inputs.shape[0]) * (self.xmax[idx] - self.xmin[idx]) + self.xmin[idx]

    def joint_sample(self, n: int) -> np.ndarray:
        d = self.xmin.shape[0]
        scale = np.diag(self.xmax - self.xmin)
        lower = np.ones((n, d)) @ np.diag(self.xmin)
        return np.random.rand(n, d) @ scale + lower


class MultivariateNormal(Distribution):

    def __init__(self, mu: np.ndarray, sigma: np.ndarray) -> None:
        self.mu = mu
        self.Sigma = sigma

    def conditional_sample(self, inputs: np.ndarray, idx: int) -> np.ndarray:
        n, d = inputs.shape
        ind = list(np.arange(d))
        Z_ind = ind[:idx] + ind[(idx + 1):]
        mu_cond = self.mu[idx] + self.Sigma[[idx]][:, Z_ind] @ np.linalg.inv(self.Sigma[Z_ind][:, Z_ind]) @ (inputs[:, Z_ind] - self.mu[[Z_ind]]).T
        mu_cond = mu_cond.flatten()
        Sigma_cond = self.Sigma[idx, idx] - (self.Sigma[[idx]][:, Z_ind] @ np.linalg.inv(self.Sigma[Z_ind][:, Z_ind]) @ self.Sigma[Z_ind][:, [idx]])
        sigma_cond = np.sqrt(Sigma_cond.item())
        return np.random.normal(mu_cond, sigma_cond, n)

    def joint_sample(self, n: int) -> np.ndarray:
        return np.random.multivariate_normal(self.mu, self.Sigma, n)
