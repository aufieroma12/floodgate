import numpy as np
from scipy.stats import norm
from util import get_knockoffs

def floodgate(inputs, f, xmin, xmax, K=50, alpha=0.05, batch_size=1, fstar=None, Y=None, ind=None):
    if fstar is None and Y is None:
        raise ValueError("Must provide either fstar of a set of evaluations from it.")
    elif Y is None:
        Y = fstar.predict(inputs)

    if isinstance(inputs, tuple):
        n, d = inputs[0].shape
    else:
        n, d = inputs.shape

    ind_full = list(np.arange(d))
    if ind is None:
        ind = ind_full

    z = norm.ppf(1 - alpha / 2)
    
    L_vals = []
    U_vals = []

    # Get surrogate predictions for original data
    Y_preds = f.predict(inputs)

    # MSE(f)
    M = (Y - Y_preds) ** 2
    M = np.mean(M.reshape(-1, batch_size), axis=1)
    M_bar = np.mean(M)
    n_batches = M.shape[0]

    # Var(Y)
    V = (n / (n - 1)) * (Y - np.mean(Y)) ** 2
    V = np.mean(V.reshape(-1, batch_size), axis=1)
    V_bar = np.mean(V)
    
    for X_ind in ind:
        knockoffs = get_knockoffs(inputs, X_ind, xmin, xmax, K)
        F = f.predict(knockoffs).reshape(n, K)
        F = np.mean(F, axis=1)
        
        # Upper bound
        Mj = ((Y - F) ** 2) - ((Y_preds - F) ** 2 / (K + 1))
        Mj = np.mean(Mj.reshape(-1, batch_size), axis=1)
        Mj_bar = np.mean(Mj)
        
        Sig = np.cov(np.vstack((Mj,V)))
        s_upper = np.sqrt(Sig[0,0] - 2 * (Mj_bar / V_bar) * Sig[0,1] + ((Mj_bar / V_bar) ** 2) * Sig[1,1]) / V_bar
        U_vals.append(min(1, Mj_bar / V_bar + z * (s_upper / np.sqrt(n_batches))))
        
        # Lower bound
        Sig = np.cov(np.vstack((Mj,M,V)))
        s_lower = np.sqrt(Sig[0,0] + Sig[1,1] + (((Mj_bar - M_bar) / V_bar) ** 2) * Sig[2,2] - 2 * Sig[0,1] + 2 * ((Mj_bar - M_bar) / V_bar) * (Sig[1,2] - Sig[0,2])) / V_bar
        L_vals.append(max(0, (Mj_bar - M_bar) / V_bar - z * (s_lower / np.sqrt(n_batches))))

    return list(zip(L_vals, U_vals))



def SPF(inputs, fstar, xmin, xmax, alpha=0.05, batch_size=1, Y=None, ind=None):
    if Y is None:
        Y = fstar.predict(inputs)

    if isinstance(inputs, tuple):
        n, d = inputs[0].shape
    else:
        n, d = inputs.shape

    ind_full = list(np.arange(d))
    if ind is None:
        ind = ind_full

    z = norm.ppf(1 - alpha / 2)
    
    L_vals = []
    U_vals = []

    # Var(Y)
    V = (n / (n - 1)) * (Y - np.mean(Y)) ** 2
    V = np.mean(V.reshape(-1, batch_size), axis=1)
    V_bar = np.mean(V)
    n_batches = V.shape[0]

    for X_ind in ind:
        # Get surrogate predictions for knockoffs
        knockoffs = get_knockoffs(inputs, X_ind, xmin, xmax, 1)
        F = fstar.predict(knockoffs)
        
        # Bounds
        Mj = ((Y - F) ** 2) / 2
        Mj = np.mean(Mj.reshape(-1, batch_size), axis=1)
        Mj_bar = np.mean(Mj)
        
        Sig = np.cov(np.vstack((Mj,V)))
        s = np.sqrt((Sig[0,0] - 2 * (Mj_bar / V_bar) * Sig[0,1] + ((Mj_bar / V_bar) ** 2) * Sig[1,1]) / (V_bar ** 2))
        U_vals.append(min(1, Mj_bar / V_bar + z * (s / np.sqrt(n_batches))))
        L_vals.append(max(0, Mj_bar / V_bar - z * (s / np.sqrt(n_batches))))

    return list(zip(L_vals, U_vals))


def panin_bound(Mjf, Vf, M, V, z):
    n = Mjf.shape[0]
    Sig = np.cov(np.vstack((Mjf,Vf,M,V)))
    Mjf = np.mean(Mjf)
    Vf = np.mean(Vf)
    M = np.mean(M)
    V = np.mean(V)

    if V >= Vf:
        E = np.sqrt(M / V)
        bound1 = E
        bound2 = E ** 2 + 2 * E * np.sqrt(Mjf / Vf)
        bound3 = E ** 2 + 2 * E * np.sqrt(1 - Mjf / Vf)

        if bound1 <= bound2 and bound1 <= bound3:
            bound = bound1
            grad_upper = np.array([1 / Vf,
                                   -Mjf / (Vf ** 2),
                                   1 / (2 * np.sqrt(M * V)),
                                   -np.sqrt(M) / (2 * V ** 1.5)])
            grad_lower = np.array([1 / Vf,
                                   -Mjf / (Vf ** 2),
                                   -1 / (2 * np.sqrt(M * V)),
                                   np.sqrt(M) / (2 * V ** 1.5)])

        elif bound2 <= bound3:
            bound = bound2
            grad_upper = np.array([1 / Vf + np.sqrt(M / (Mjf * Vf * V)),
                                   -Mjf / (Vf ** 2) - np.sqrt(Mjf * M / (V * (Vf ** 3))),
                                   1 / V + np.sqrt(Mjf / (M * Vf * V)),
                                   -M / (V ** 2) - np.sqrt(Mjf * M / (Vf * (V ** 3)))])
            grad_lower = np.array([1 / Vf - np.sqrt(M / (Mjf * Vf * V)),
                                   -Mjf / (Vf ** 2) + np.sqrt(Mjf * M / (V * (Vf ** 3))),
                                   -1 / V - np.sqrt(Mjf / (M * Vf * V)),
                                   M / (V ** 2) + np.sqrt(Mjf * M / (Vf * (V ** 3)))])

        else:
            bound = bound3
            grad_upper = np.array([1 / Vf - np.sqrt(M / V) * (1 - Mjf / Vf) ** (-.5) / Vf,
                                   -Mjf / (Vf ** 2) + np.sqrt(M / V) * (1 - Mjf / Vf) ** (-.5) * (Mjf / Vf**2),
                                   1 / V + 2 / np.sqrt(M * V) * np.sqrt(1 - Mjf / Vf),
                                   -M / (V ** 2) - np.sqrt((M / (V ** 3)) * (1 - Mjf / Vf))])
            grad_lower = np.array([1 / Vf + np.sqrt(M / V) * (1 - Mjf / Vf)**(-.5) / Vf,
                                   -Mjf / (Vf ** 2) - np.sqrt(M / V) * (1 - Mjf / Vf) ** (-.5) * (Mjf / Vf**2),
                                   -1 / V - 2 / np.sqrt(M * V) * np.sqrt(1 - Mjf / Vf),
                                   M / (V ** 2) + np.sqrt((M / (V ** 3)) * (1 - Mjf / Vf))])

    else:
        Sig = Sig[:3, :3]
        E = np.sqrt(M / Vf)
        bound1 = E
        bound2 = E ** 2 + 2 * E * np.sqrt(Mjf / Vf)
        bound3 = E ** 2 + 2 * E * np.sqrt(1 - Mjf / Vf)

        if bound1 <= bound2 and bound1 <= bound3:
            bound = bound1
            grad_upper = np.array([1 / Vf,
                                   -Mjf / (Vf ** 2) - .5 * np.sqrt(M / (Vf ** 3)),
                                   1 / (2 * np.sqrt(M * Vf))])
            grad_lower = np.array([1 / Vf,
                                   -Mjf / (Vf ** 2) + .5 * np.sqrt(M / (Vf ** 3)),
                                   -1 / (2 * np.sqrt(M * Vf))])

        elif bound2 <= bound3:
            bound = bound2
            grad_upper = np.array([(1 + np.sqrt(M / Mjf)) / Vf,
                                   -(Mjf + M + 2 * np.sqrt(Mjf * M)) / (Vf ** 2),
                                   (1 + np.sqrt(Mjf / M)) / Vf])
            grad_lower = np.array([(1 - np.sqrt(M / Mjf)) / Vf,
                                   -(Mjf - M - 2 * np.sqrt(Mjf * M)) / (Vf ** 2),
                                   -(1 + np.sqrt(Mjf / M)) / Vf])

        else:
            bound = bound3
            grad_upper = np.array([1 / Vf - (M / Vf - Mjf * M / (Vf ** 2)) ** (-.5) * (M / (Vf ** 2)),
                                   -(Mjf + M) / (Vf ** 2) + (M / Vf - Mjf * M / (Vf ** 2)) ** (-.5) * (-M / (Vf ** 2) + 2 * Mjf * M / (Vf ** 3)),
                                   1 / Vf + (M / Vf - Mjf * M / (Vf ** 2)) ** (-.5) * (1 / Vf - Mjf / (Vf ** 2))])
            grad_lower = np.array([1 / Vf + (M / Vf - Mjf * M / (Vf ** 2)) ** (-.5) * (M / (Vf ** 2)),
                                   -(Mjf + M) / (Vf ** 2) - (M / Vf - Mjf * M / (Vf ** 2)) ** (-.5) * (-M / (Vf ** 2) + 2 * Mjf * M / (Vf ** 3)),
                                   1 / Vf - (M / Vf - Mjf * M / (Vf ** 2)) ** (-.5) * (1 / Vf - Mjf / (Vf ** 2))])
 
    s_upper = grad_upper.T @ Sig @ grad_upper
    s_lower = grad_lower.T @ Sig @ grad_lower
    Lower = max(0, Mjf / Vf - bound - z * (s_lower/np.sqrt(n)))
    Upper = min(1, Mjf / Vf + bound + z * (s_upper/np.sqrt(n)))
    return (Lower, Upper)



def combined_surrogate_methods(inputs, f, xmin, xmax, K=50, alpha=0.05, batch_size=1, fstar=None, Y=None, ind=None):
    if fstar is None and Y is None:
        raise ValueError("Must provide either fstar of a set of evaluations from it.")
    elif Y is None:
        Y = fstar.predict(inputs)

    if isinstance(inputs, tuple):
        n, d = inputs[0].shape
    else:
        n, d = inputs.shape   

    ind_full = list(np.arange(d))
    if ind is None:
        ind = ind_full

    z = norm.ppf(1 - alpha / 2)

    L_vals_floodgate = []
    L_vals_spf = []
    L_vals_panin = []
    U_vals_floodgate = []
    U_vals_spf = []
    U_vals_panin = []

    # Get surrogate predictions for original data
    Y_preds = f.predict(inputs)

    # MSE(f)
    M = (Y - Y_preds) ** 2
    M = np.mean(M.reshape(-1, batch_size), axis=1)
    M_bar = np.mean(M)
    n_batches = M.shape[0]

    # Var(Y)
    V = (n / (n - 1)) * (Y - np.mean(Y)) ** 2
    V = np.mean(V.reshape(-1, batch_size), axis=1)
    V_bar = np.mean(V)
    
    for X_ind in ind:
        knockoffs = get_knockoffs(inputs, X_ind, xmin, xmax, K)
        F = f.predict(knockoffs).reshape(n, K)
        F1 = F[:,0]
        F = np.mean(F, axis=1)
        
        # Upper bound floodgate
        Mj = ((Y - F) ** 2) - ((Y_preds - F) ** 2 / (K + 1))
        Mj = np.mean(Mj.reshape(-1, batch_size), axis=1)
        Mj_bar = np.mean(Mj)
        
        Sig = np.cov(np.vstack((Mj,V)))
        s_upper = np.sqrt(Sig[0,0] - 2 * (Mj_bar / V_bar) * Sig[0,1] + ((Mj_bar / V_bar) ** 2) * Sig[1,1]) / V_bar
        U_vals_floodgate.append(min(1, Mj_bar / V_bar + z * (s_upper / np.sqrt(n_batches))))
        
        # Lower bound floodgate
        Sig = np.cov(np.vstack((Mj,M,V)))
        s_lower = np.sqrt(Sig[0,0] + Sig[1,1] + (((Mj_bar - M_bar) / V_bar) ** 2) * Sig[2,2] - 2 * Sig[0,1] + 2 * ((Mj_bar - M_bar) / V_bar) * (Sig[1,2] - Sig[0,2])) / V_bar
        L_vals_floodgate.append(max(0, (Mj_bar - M_bar) / V_bar - z * (s_lower / np.sqrt(n_batches))))


        # Bounds SPF
        Vf = (n / (n - 1)) * (Y_preds - np.mean(Y_preds)) ** 2
        Vf = np.mean(Vf.reshape(-1, batch_size), axis=1)
        Vf_bar = np.mean(Vf)

        Mjf = ((Y_preds - F1) ** 2) / 2
        Mjf = np.mean(Mjf.reshape(-1, batch_size), axis=1)
        Mjf_bar = np.mean(Mjf)
        
        Sig = np.cov(np.vstack((Mjf,Vf)))
        s = np.sqrt(Sig[0,0] - 2 * (Mjf_bar / Vf_bar) * Sig[0,1] + ((Mjf_bar / Vf_bar) ** 2) * Sig[1,1]) / Vf_bar
        L_vals_spf.append(max(0, Mjf_bar / Vf_bar - z * (s / np.sqrt(n_batches))))
        U_vals_spf.append(min(1, Mjf_bar / Vf_bar + z * (s / np.sqrt(n_batches))))
  

        # Bounds Panin
        bounds = panin_bound(Mjf, Vf, M, V, z)
        L_vals_panin.append(bounds[0])
        U_vals_panin.append(bounds[1])

    floodgate_ret = list(zip(L_vals_floodgate, U_vals_floodgate))
    spf_ret = list(zip(L_vals_spf, U_vals_spf))
    panin_ret = list(zip(L_vals_panin, U_vals_panin))

    return floodgate_ret, spf_ret, panin_ret

