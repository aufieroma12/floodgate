import numpy as np

KRR_hyperparams = {
	100: (np.logspace(-8, 3, 30), np.logspace(-5, 4, 20)),
	1000: (np.logspace(-10, 2, 20), np.logspace(-5, 4, 20)),
	10000: (np.logspace(-13, -10, 4), np.logspace(-4, -2, 3)),
	50000: (np.logspace(-12, -10, 3), np.logspace(-4, -2, 3)),
	100000: (np.logspace(-12, -10, 3), np.logspace(-4, -2, 3))
}

Hymod_inputs = {
	"labels": ['Sm', 'beta', 'alfa', 'Rs', 'Rf'],
	"min": np.array([0, 0, 0, 0, 0.1]),
	"max": np.array([400, 2, 1, 0.1, 1])
}