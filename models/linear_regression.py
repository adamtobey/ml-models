import numpy as np
from linear_basis_functions import ScalarBasisFunctions

class LinearRegression(object):

    def __init__(self, basis_function=ScalarBasisFunctions.Identity(), l2_cost=None):
        self.weights = None
        self.basis_function = basis_function
        self.l2_cost = l2_cost

    def regularize(self, X, y):
        if self.l2_cost is None:
            return X, y
        else:
            rl2 = self.l2_cost ** 0.5
            n_weights = X.shape[1]
            t_in = np.concatenate([X, np.diag([rl2 for _ in range(n_weights)])])
            t_out = np.concatenate([y, [0 for _ in range(n_weights)]])
            return t_in, t_out

    def fit(self, X, y):
        X = self.basis_function(X)
        X, y = self.regularize(X, y)
        self.weights = np.linalg.lstsq(X, y)[0]

    def predict(self, X):
        assert self.weights is not None, "Model must be trained before predicting"
        X = self.basis_function(X)
        return X.dot(self.weights)
