import numpy as np
from optimizers import Function, MomentumOptimizer
from linear_basis_functions import BasisFunctions

def sigmoid(x):
    def sig(x):
        return 1 / (1 + np.exp(-x))
    out = np.ndarray(x.size)
    out[x > 0] = sig(x[x > 0])
    out[x < 0] = 1 - sig(-x[x < 0])
    return out

class LogisticRegressionCost(Function):

    def eval(self, X, t, w):
        affine = X.dot(w)
        z = 2 * t - 1
        return -np.sum(np.log(sigmoid(z * affine)), axis=0)

    def gradient(self, X, t, w):
        z = 2 * t - 1
        sn = sigmoid(z * X.dot(w))
        return -np.sum(((1 - sn) * z).reshape(-1, 1) * X, axis=0)

    # for Newton's method -- have to be careful about this since
    # the cost function is nearly linear at the extrema...
    def hessian(self, X, t, w):
        def single_hessian(x, t, w):
            z = 2 * t - 1
            a = z * w.dot(x)
            if a < 0:
                B = np.exp(a) + 1
                b = np.log(B)
                A = np.exp(a - 2 * b)
                diag = np.ones(x.shape) * z * (A + B**-2)
                return np.outer(x, x) * A + np.diag(diag)
            else:
                epsilon = 1e-6 # avoid dividing by zero
                A = np.exp(-a)
                diag = np.ones(x.shape) * z * (A**2 + A + epsilon) / ((1 + A)**2 + epsilon)
                return np.outer(x, x) * A / ((1 + A)**2 + epsilon) + np.diag(diag)
        return sum(single_hessian(x, t, w) for x, t in zip(X, t))

class LogisticRegression(object):

    def __init__(self, basis_function=BasisFunctions.Affine(), optimizer=MomentumOptimizer(learning_rate=1e-2, num_epochs=1000)):
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
        return sigmoid(self.basis_function(X).dot(self.weights))
