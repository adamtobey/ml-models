import numpy as np
from optimizers import NewtonsMethod, Function
from linear_basis_functions import BasisFunctions

class LogisticRegressionCost(Function):

    def eval(self, X, t, w):
        affine = X.dot(w)
        z = 2 * t - 1
        return np.sum(np.log(1 + z * affine), axis=0)

    def gradient(self, X, t, w):
        z = 2 * t - 1
        a = z * X.dot(w)
        return -np.sum((1 / (np.exp(-a) + 1) * z).reshape(-1, 1) * X, axis=0)

    def hessian(self, X, t, w):
        def single_hessian(x, t, w):
            z = 2 * t - 1
            a = -z * w.dot(x)
            B = np.exp(-a) + 1
            b = np.log(B)
            A = np.exp(-a - 2 * b)
            diag = np.ones(x.shape) * z * (A + B**-2)
            return np.outer(x, x) * A + np.diag(diag)
        return sum(single_hessian(x, t, w) for x, t in zip(X, t))

class RefLRC(Function):

    def sigmoid(self, x):
        return 1 / (1 + np.exp(x))

    def gradient(self, X, t, w):
        z = 2 * t - 1
        sn = self.sigmoid(z * X.dot(w))
        return -np.sum(((1 - sn) * z).reshape(-1, 1) * X, axis=0)

    def hessian(self, X, t, w):
        epsilon = 1e-6 # avoid dividing by zero
        def single_hessian(x, t, w):
            z = 2 * t - 1
            a = np.exp(-z * w.dot(x)) # TODO sometimes explodes
            diag = np.ones(x.shape) * z * (a**2 + a + epsilon) / ((1 + a)**2 + epsilon)
            return np.outer(x, x) * a / ((1 + a)**2 + epsilon) + np.diag(diag)
        return sum(single_hessian(x, t, w) for x, t in zip(X, t))

class LogisticRegression(object):

    def __init__(self, basis_function=BasisFunctions.Affine(), optimizer=NewtonsMethod(5e-3)):
        self.optimizer = optimizer
        self.weights = None
        self.weights_init = lambda d: np.random.rand(d) * 0.1
        self.basis_function = basis_function
        self.cost = LogisticRegressionCost()

    def fit(self, X, y, num_epochs):
        X = self.basis_function(X)
        self.weights = self.optimizer.opt(self.cost, X, y, self.weights_init(X.shape[1]))

    def predict(self, X):
        assert self.weights is not None, "Model must be trained before predicting"
        return self.sigmoid(self.basis_function(X).dot(self.weights))
