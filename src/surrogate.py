from abc import ABC, abstractmethod

import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.model_selection import GridSearchCV
from safepython.model_execution import model_execution
from safepython import HyMod

from config import Hymod_inputs


def batched(batch_size):
    def f_batched(f):
        def wrapper(*args, **kwargs):
            inputs = args[1]
            if isinstance(inputs, tuple):
                X, met = inputs
            else:
                X = inputs
            n = X.shape[0]
            if n <= batch_size:
                return f(*args, **kwargs)
            else:
                y_preds = []
                for i in range(n // batch_size + (n % batch_size > 0)):
                    X_batch = X[i*batch_size:(i+1)*batch_size]
                    if isinstance(inputs, tuple):
                        met_batch = met[i*batch_size:(i+1)*batch_size]
                        X_batch = (X_batch, met_batch)
                    y_preds.append(f(args[0], X_batch))
                return np.concatenate(y_preds, axis=0)
        return wrapper
    return f_batched



class Surrogate(ABC):
    def __init__(self, model):
        self.model = model    
    
    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError("self.fit() not implemented")
        
    @abstractmethod
    def predict(self, X):
        raise NotImplementedError("self.predict() not implemented")



class Hymod(Surrogate):
    def __init__(self, obsPath=Hymod_inputs["FORCING_PATH"]):
        # Observed inputs and outputs
        data = np.genfromtxt(obsPath, comments='%')
        rain = data[0:365, 0] # 2-year simulation
        evap = data[0:365, 1]
        flow = data[0:365, 2]
        warmup = 30

        super().__init__(lambda X: model_execution(HyMod.hymod_nse, X, rain, evap, flow, warmup)[:,0])

    def fit(self, X, y):
        pass

    def predict(self, X):
        return self.model(X)


class Ishigami(Surrogate):

    def __init__(self, a = 1, b = 7, c = 0.1):
        super().__init__(None)
        self.a = a
        self.b = b
        self.c = c

    def fit(self, X, y):
        pass

    def predict(self, X):
        return (
            self.a * np.sin(X[:,0]) +
            self.b * np.sin(X[:,1])**2 +
            self.c * X[:,2]**4 * np.sin(X[:,0])
        )


class LinReg(Surrogate):
    def __init__(self):
        super().__init__(LinearRegression())
    
    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
    


class PolyReg(Surrogate):
    def __init__(self, degree):
        super().__init__(LinearRegression(fit_intercept=False))
        self.degree = degree
        self.poly = PolynomialFeatures(degree)
        
    def fit(self, X, y):
        X = self.poly.fit_transform(X)
        self.model.fit(X, y)
        
    def predict(self, X):
        X = self.poly.transform(X)
        return self.model.predict(X)
    


class PolyRegLASSOcv(Surrogate):
    def __init__(self, degree):
        super().__init__(LassoCV(fit_intercept=False, alphas=np.logspace(-10, 5, 100), max_iter=2500))
        self.degree = degree
        self.poly = PolynomialFeatures(degree)
    
    def fit(self, X, y):
        X = self.poly.fit_transform(X)
        self.model.fit(X, y)        
        self.alpha_ = self.model.alpha_
        self.coef_ = self.model.coef_
        self.alphas_ = self.model.alphas_
        
    def predict(self, X):
        X = self.poly.transform(X)
        return self.model.predict(X)
        


class KRRcv(Surrogate):
    def __init__(self, alphas, gammas):
        base_model = KernelRidge(kernel='rbf')
        parameters = {'alpha': alphas, 'gamma': gammas}
        super().__init__(GridSearchCV(base_model, parameters, scoring='neg_mean_squared_error'))

    def fit(self, X, y):
        self.model.fit(X, y)

    @batched(int(5e4))
    def predict(self, X):
        return self.model.predict(X)
    


class KNNcv(Surrogate):
    def __init__(self, neighbors):
        base_model = KNN()
        weights = ['uniform', 'distance']
        parameters = {'n_neighbors': neighbors, 'weights': weights}
        super().__init__(GridSearchCV(base_model, parameters, scoring='neg_mean_squared_error'))

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
