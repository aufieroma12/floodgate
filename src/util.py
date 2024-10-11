import numpy as np

def get_knockoffs(inputs, X_ind, xmin, xmax, K):
    if isinstance(inputs, tuple):
        X, met = inputs 
        met_new = np.zeros((met.shape[0]*K, met.shape[1], met.shape[2]))
    else:
        X = inputs

    n, d = X.shape
    ind = list(np.arange(d))
    Z_ind = ind[:X_ind] + ind[(X_ind+1):]

    knockoffs = np.zeros(((n*K), d))
    knockoffs[:,X_ind] = np.random.rand(n*K) * (xmax[X_ind] - xmin[X_ind]) + xmin[X_ind]
    for i in range(n):
        knockoffs[(i*K):((i+1)*K), Z_ind] = X[i, Z_ind]
    if isinstance(inputs, tuple):
        for i in range(n):
            met_new[(i*K):((i+1)*K),:] = met[i,:,:]
        knockoffs = (knockoffs, met_new)

    return knockoffs


def conditional_sample(mu, Sigma, inputs, X_ind):
    n, d = inputs.shape
    ind = list(np.arange(d))
    Z_ind = ind[:X_ind] + ind[(X_ind+1):]
    mu_cond = mu[X_ind] + Sigma[[X_ind]][:, Z_ind] @ np.linalg.inv(Sigma[Z_ind][:,Z_ind]) @ (inputs[:, Z_ind] - mu[[Z_ind]]).T
    mu_cond = mu_cond.flatten()
    Sigma_cond = Sigma[X_ind, X_ind] - (Sigma[[X_ind]][:, Z_ind] @ np.linalg.inv(Sigma[Z_ind][:,Z_ind]) @ Sigma[Z_ind][:, [X_ind]])
    sigma_cond = np.sqrt(Sigma_cond.item())
    return np.random.normal(mu_cond, sigma_cond, n)


def get_knockoffs_normal(inputs, X_ind, mu, Sigma, K):
    n, d = inputs.shape
    ind = list(np.arange(d))
    Z_ind = ind[:X_ind] + ind[(X_ind+1):]

    knockoffs = np.zeros(((n*K), d))
    for i in range(n):
        knockoffs[(i*K):((i+1)*K), Z_ind] = inputs[i, Z_ind]

    knockoffs[:, X_ind] = conditional_sample(mu, Sigma, knockoffs, X_ind)
    return knockoffs
