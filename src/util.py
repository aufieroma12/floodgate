import numpy as np

def get_knockoffs(inputs, X_ind, xmin, xmax, K):
    if isinstance(inputs, tuple):
        X, met = inputs
        met_new = np.zeros(((n*K), met.shape[1], met.shape[2]))
    else:
        X = inputs

    n, d = X.shape
    ind = list(np.arange(d))
    Z_ind = ind[:X_ind] + ind[(X_ind+1):]

    knockoffs = np.zeros(((n*K), d))
    knockoffs[:,X_ind] = np.random.rand(n*K) * (xmax[X_ind] - xmin[X_ind]) + xmin[X_ind]
    for i in range(n):
        knockoffs[(i*K):((i+1)*K),Z_ind] = X[i, Z_ind]
    if isinstance(inputs, tuple):
        for i in range(n):
            met_new[(i*K):((i+1)*K),:] = met[i,:,:]
        knockoffs = (knockoffs, met_new)

    return knockoffs