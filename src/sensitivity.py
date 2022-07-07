from scipy.stats import norm

def floodgate(X, f, xmin, xmax, K=50, alpha=0.05, fstar=None, Y=None, ind=None, batch_size=None):
    if fstar is None and Y is None:
        raise ValueError("Must provide either fstar of a set of evaluations from it.")
    elif Y is None:
        Y = fstar.predict(X, batch_size=batch_size)

    (n, d) = X.shape
    if batch_size is None:
        batch_size = n    

    ind_full = list(np.arange(d))
    if ind is None:
        ind = ind_full

    z = norm.ppf(1 - alpha / 2)
    
    L_vals = []
    U_vals = []

    # Get surrogate predictions for original data
    Y_preds = f.predict(X, batch_size=batch_size)

    M = (Y - Y_preds) ** 2
    M_bar = np.mean(M)
    
    for X_ind in ind:
        Z_ind = ind_full[:X_ind] + ind_full[(X_ind+1):]
        
        # Get surrogate predictions for knockoffs
        knockoffs = np.zeros(((n*K), d))

        for i in range(n):
            knockoffs[(i*K):((i+1)*K),X_ind] = np.random.rand(K) * (xmax[X_ind]-xmin[X_ind]) + xmin[X_ind]
            knockoffs[(i*K):((i+1)*K),Z_ind] = X[i, Z_ind]
        F = f.predict(knockoffs, batch_size=batch_size).reshape(n, K)
        F = np.mean(F, axis=1)
        
        # Var(Y)
        V = (n / (n - 1)) * (Y - np.mean(Y))**2
        V_bar = np.mean(V)
        
        # Upper bound
        Mj = ((Y - F) ** 2) - ((Y_preds - F) ** 2 / (K + 1))
        Mj_bar = np.mean(Mj)
        
        Sig = np.cov(np.vstack((Mj,V)))
        s_upper = np.sqrt(Sig[0,0] - 2 * (Mj_bar / V_bar) * Sig[0,1] + ((Mj_bar / V_bar) ** 2) * Sig[1,1]) / V_bar

        U_vals.append(min(1, Mj_bar / V_bar + z * (s_upper / np.sqrt(n))))
        
        # Lower bound
        Sig = np.cov(np.vstack((Mj,M,V)))
        s_lower = np.sqrt(Sig[0,0] + Sig[1,1] + (((Mj_bar - M_bar) / V_bar) ** 2) * Sig[2,2] - 2 * Sig[0,1] + 2 * ((Mj_bar - M_bar) / V_bar) * (Sig[1,2] - Sig[0,2])) / V_bar
        
        L_vals.append(max(0, (Mj_bar - M_bar) / V_bar - z * (s_lower / np.sqrt(n))))

    return list(zip(L_vals, U_vals))


def SPF(X, xmin, xmax, fstar, Y=None, alpha=0.05, ind=None, batch_size=None):
    elif Y is None:
        Y = fstar.predict(X, batch_size=batch_size)

    (n, d) = X.shape
    if batch_size is None:
        batch_size = n

    ind_full = list(np.arange(d))
    if ind is None:
        ind = ind_full

    z = norm.ppf(1 - alpha / 2)
    
    L_vals = []
    U_vals = []
    
    for X_ind in ind:
        Z_ind = ind[:X_ind] + ind[(X_ind+1):]
        
        # Get surrogate predictions for knockoffs
        knockoffs = X
        knockoffs[:,X_ind] = np.random.rand(n) * (xmax[X_ind] - xmin[X_ind]) + xmin[X_ind]
        F = fstar.predict(knockoffs, batch_size=batch_size)
        
        # Var(Y)
        V = (n / (n - 1)) * (Y - np.mean(Y)) ** 2
        V_bar = np.mean(V)
        
        # Bounds
        Mj = ((Y - F) ** 2) / 2
        Mj_bar = np.mean(M)
        
        Sig = np.cov(np.vstack((Mj,V)))
        s = np.sqrt((Sig[0,0] - 2 * (Mj_bar / V_bar) * Sig[0,1] + ((Mj_bar / V_bar) ** 2) * Sig[1,1]) / (V_bar ** 2))
        
        U_vals.append(Mj_bar / V_bar + z * (s / np.sqrt(n)))
        L_vals.append(Mj_bar / V_bar - z * (s / np.sqrt(n)))

    return list(zip(L_vals, U_vals))


def panin_bound(Mjf, Vf, Mj, V, z):
    n = Mf.shape[0]
    Sig = np.cov(np.vstack((Mjf,Vf,Mj,V)))
    Mjf = np.mean(Mjf)
    Vf = np.mean(Vf)
    Mj = np.mean(Mj)
    V = np.mean(V)

    if V >= Vf:
        E = np.sqrt(Mj / V)
        bound1 = E
        bound2 = E ** 2 + 2 * E * np.sqrt(Mjf / Vf)
        bound3 = E ** 2 + 2 * E * np.sqrt(1 - Mjf / Vf)

        if bound1 <= bound2 and bound1 <= bound3:
            bound = bound1
            grad_upper = np.array([1 / Vf,
                                   -Mjf / (Vf ** 2),
                                   1 / (2 * np.sqrt(Mj * V)),
                                   -np.sqrt(Mj) / (2 * V ** 1.5)])
            grad_lower = np.array([1 / Vf,
                                   -Mjf / (Vf ** 2),
                                   -1 / (2 * np.sqrt(Mj * V)),
                                   np.sqrt(Mj) / (2 * V ** 1.5)])

        elif bound2 <= bound3:
            bound = bound2
            grad_upper = np.array([1 / Vf + np.sqrt(Mj / (Mjf * Vf * V)),
                                   -Mjf / (Vf ** 2) - np.sqrt(Mjf * Mj / (V * (Vf ** 3))),
                                   1 / V + np.sqrt(Mjf / (Mj * Vf * V)),
                                   -Mj / (V ** 2) - np.sqrt(Mjf * Mj / (Vf * (V ** 3)))])
            grad_lower = np.array([1 / Vf - np.sqrt(Mj / (Mjf * Vf * V)),
                                   -Mjf / (Vf ** 2) + np.sqrt(Mjf * Mj / (V * (Vf ** 3))),
                                   -1 / V - np.sqrt(Mjf / (Mj * Vf * V)),
                                   Mj / (V ** 2) + np.sqrt(Mjf * Mj / (Vf * (V ** 3)))])

        else:
            bound = bound3
            grad_upper = np.array([1 / Vf - np.sqrt(Mj / V) * (1 - Mjf / Vf) ** (-.5) / Vf,
                                   -Mjf / (Vf ** 2) + np.sqrt(Mj / V) * (1 - Mjf / Vf) ** (-.5) * (Mjf / Vf**2),
                                   1 / V + 2 / np.sqrt(Mj * V) * np.sqrt(1 - Mjf / Vf),
                                   -Mj / (V ** 2) - np.sqrt((Mj / (V ** 3)) * (1 - Mjf / Vf))])
            grad_lower = np.array([1 / Vf + np.sqrt(Mj / V) * (1 - Mjf / Vf)**(-.5) / Vf,
                                   -Mjf / (Vf ** 2) - np.sqrt(Mj / V) * (1 - Mjf / Vf) ** (-.5) * (Mjf / Vf**2),
                                   -1 / V - 2 / np.sqrt(Mj * V) * np.sqrt(1 - Mjf / Vf),
                                   Mj / (V ** 2) + np.sqrt((Mj / (V ** 3)) * (1 - Mjf / Vf))])

    else:
        Sig = Sig[:3, :3]
        E = np.sqrt(Mj / Vf)
        bound1 = E
        bound2 = E ** 2 + 2 * E * np.sqrt(Mjf / Vf)
        bound3 = E ** 2 + 2 * E * np.sqrt(1 - Mjf / Vf)

        if bound1 <= bound2 and bound1 <= bound3:
            bound = bound1
            grad_upper = np.array([1 / Vf,
                                   -Mjf / (Vf ** 2) - .5 * np.sqrt(Mj / (Vf ** 3)),
                                   1 / (2 * np.sqrt(Mj * Vf))])
            grad_lower = np.array([1 / Vf,
                                   -Mjf / (Vf ** 2) + .5 * np.sqrt(Mj / (Vf ** 3)),
                                   -1 / (2 * np.sqrt(Mj * Vf))])

        elif bound2 <= bound3:
            bound = bound2
            grad_upper = np.array([(1 + np.sqrt(Mj / Mjf)) / Vf,
                                   -(Mjf + Mj + 2 * np.sqrt(Mjf * Mj)) / (Vf ** 2),
                                   (1 + np.sqrt(Mjf / Mj)) / Vf])
            grad_lower = np.array([(1 - np.sqrt(Mj / Mjf)) / Vf,
                                   -(Mjf - Mj - 2 * np.sqrt(Mjf * Mj)) / (Vf ** 2),
                                   -(1 + np.sqrt(Mjf / Mj)) / Vf])

        else:
            bound = bound3
            grad_upper = np.array([1 / Vf - (Mj / Vf - Mjf * Mj / (Vf ** 2)) ** (-.5) * (Mj / (Vf ** 2)),
                                   -(Mjf + Mj) / (Vf ** 2) + (Mj / Vf - Mjf * Mj / (Vf ** 2)) ** (-.5) * (-Mj / (Vf ** 2) + 2 * Mjf * Mj / (Vf ** 3)),
                                   1 / Vf + (Mj / Vf - Mjf * Mj / (Vf ** 2)) ** (-.5) * (1 / Vf - Mjf / (Vf ** 2))])
            grad_lower = np.array([1 / Vf + (Mj / Vf - Mjf * M / (Vf ** 2)) ** (-.5) * (Mj / (Vf ** 2)),
                                   -(Mjf + Mj) / (Vf ** 2) - (Mj / Vf - Mjf * Mj / (Vf ** 2)) ** (-.5) * (-Mj / (Vf ** 2) + 2 * Mjf * Mj / (Vf ** 3)),
                                   1 / Vf - (Mj / Vf - Mjf * Mj / (Vf ** 2)) ** (-.5) * (1 / Vf - Mjf / (Vf ** 2))])
 
    s_upper = grad_upper.T @ Sig @ grad_upper
    s_lower = grad_lower.T @ Sig @ grad_lower
    return (Mjf / Vf - bound - z * (s_lower/np.sqrt(n)), Mjf / Vf + bound + z * (s_upper/np.sqrt(n)))


def combined_surrogate_methods(X, f, xmin, xmax, K=50, alpha=0.05, fstar=None, Y=None, ind=None):
    if fstar is None and Y is None:
        raise ValueError("Must provide either fstar of a set of evaluations from it.")
    elif Y is None:
        Y = fstar.predict(X)

    (n, d) = X.shape
    if batch_size is None:
        batch_size = n    

    ind_full = list(np.arange(d))
    if ind is None:
        ind = ind_full

    z = norm.ppf(1 - alpha / 2)

    L_vals_floodgate = []
    L_vals_SPF = []
    L_vals_panin = []
    U_vals_floodgate = []
    U_vals_SPF = []
    U_vals_panin = []

    # Get surrogate predictions for original data
    Y_preds = f.predict(X)

    M = (Y - Y_preds) ** 2
    M_bar = np.mean(M)
    
    for X_ind in ind:
        Z_ind = ind_full[:X_ind] + ind_full[(X_ind+1):]
        
        # Get surrogate predictions for knockoffs
        knockoffs = np.zeros(((n*K), d))

        for i in range(n):
            knockoffs[(i*K):((i+1)*K),X_ind] = np.random.rand(K) * (xmax[X_ind] - xmin[X_ind]) + xmin[X_ind]
            knockoffs[(i*K):((i+1)*K),Z_ind] = X[i, Z_ind]
        F = f.predict(knockoffs).reshape(n, K)
        F1 = F[:,0]
        F = np.mean(F, axis=1)

        # Var(Y)
        V = (n / (n - 1)) * (Y - np.mean(Y)) ** 2
        V_bar = np.mean(V)
        
        # Upper bound floodgate
        Mj = ((Y - F) ** 2) - ((Y_preds - F) ** 2 / (K + 1))
        Mj_bar = np.mean(Mj)
        
        Sig = np.cov(np.vstack((Mj,V)))
        s_upper = np.sqrt(Sig[0,0] - 2 * (Mj_bar / V_bar) * Sig[0,1] + ((Mj_bar / V_bar) ** 2) * Sig[1,1]) / V_bar

        U_vals_floodgate.append(min(1, Mj_bar / V_bar + z * (s_upper / np.sqrt(n))))
        
        # Lower bound floodgate
        Sig = np.cov(np.vstack((Mj,M,V)))
        s_lower = np.sqrt(Sig[0,0] + Sig[1,1] + (((Mj_bar - M_bar) / V_bar) ** 2) * Sig[2,2] - 2 * Sig[0,1] + 2 * ((Mj_bar - M_bar) / V_bar) * (Sig[1,2] - Sig[0,2])) / V_bar
        
        L_vals_floodgate.append(max(0, (Mj_bar - M_bar) / V_bar - z * (s_lower / np.sqrt(n))))


        # Bounds surrogate
        Vf = (n / (n - 1)) * (Y_preds - np.mean(Y_preds)) ** 2
        Vf_bar = np.mean(Vf)

        Mjf = ((Y_preds - F1) ** 2) / 2
        Mjf_bar = np.mean(Mjf)
        
        Sig = np.cov(np.vstack((Mjf,Vf)))
        s = np.sqrt((Sig[0,0] - 2 * (Mjf_bar / Vf_bar) * Sig[0,1] + ((Mjf_bar / Vf_bar) ** 2) * Sig[1,1]) / (Vf_bar ** 2))
        
        U_vals_spf.append(Mjf_bar / Vf_bar + z * (s / np.sqrt(n)))
        L_vals_spf.append(Mjf_bar / Vf_bar - z * (s / np.sqrt(n)))
  

        # Bounds Panin
        bounds = panin_bound(Mjf, Vf, Mj, V, z)
        L_vals_panin.append(bounds[0])
        U_vals_panin.append(bounds[1])


    floodgate_ret = list(zip(L_vals_floodgate, U_vals_floodgate))
    spf_ret = list(zip(L_vals_spf, U_vals_spf))
    panin_ret = list(zip(L_vals_panin, U_vals_panin))

    return floodgate_ret, spf_ret, panin_ret

