import numpy as np

class ScalarBasisFunctions(object):

    @staticmethod
    def Identity():
        return lambda x: x.reshape(1, -1)

    @staticmethod
    def Polynomial(degree):
        return lambda x: np.concatenate([(x**d).reshape(1, -1) for d in range(degree + 1)]).T

class LinearRegression(object):
    def __init__(self, basis_function=lambda x: x):
        self.weights = None
        self.basis_function = basis_function

    def fit(self, inputs, targets):
        self.weights = np.linalg.lstsq(self.basis_function(inputs), targets)[0]

    def predict(self, inputs):
        assert self.weights is not None, "Model must be trained before predicting"
        return self.basis_function(inputs).dot(self.weights)
