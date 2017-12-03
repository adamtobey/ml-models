import numpy as np

class ScalarBasisFunctions(object):

    @staticmethod
    def Identity():
        return lambda x: x.reshape(1, -1)

    @staticmethod
    def Polynomial(degree):
        return lambda x: np.concatenate([(x**d).reshape(1, -1) for d in range(degree + 1)]).T

class Regularizers(object):

    @staticmethod
    def NoReg():
        return lambda *x: x

    @staticmethod
    def L2(l2_cost):
        rl2 = l2_cost ** 0.5
        def transform(inputs, targets):
            n_weights = inputs.shape[1]
            t_in = np.concatenate([inputs, np.diag([rl2 for _ in range(n_weights)])])
            t_out = np.concatenate([targets, [0 for _ in range(n_weights)]])
            return t_in, t_out
        return transform

class LinearRegression(object):
    def __init__(self, basis_function=lambda x: x, regularizer=Regularizers.NoReg()):
        self.weights = None
        self.make_basis_matrix = basis_function
        self.regularize = regularizer

    def fit(self, inputs, targets):
        inputs, targets = self.regularize(self.make_basis_matrix(inputs), targets)
        self.weights = np.linalg.lstsq(inputs, targets)[0]

    def predict(self, inputs):
        assert self.weights is not None, "Model must be trained before predicting"
        return self.make_basis_matrix(inputs).dot(self.weights)
