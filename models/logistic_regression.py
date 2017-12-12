import numpy as np
from optimizers import GradientDescentOptimizer
from linear_basis_functions import BasisFunctions

class LogisticRegression(object):

    def __init__(self, optimizer=GradientDescentOptimizer(learning_rate=0.1), basis_function=BasisFunctions.Affine()):
        self.optimizer = optimizer
        self.weights = None
        self.weights_init = lambda d: np.random.rand(d) * 0.1
        self.basis_function = basis_function

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def cost(self, batch, targets, weights):
        affine = batch.dot(weights)
        z = 2 * targets - 1
        return -np.sum(np.log(self.sigmoid(z * affine)), axis=0)

    def cost_grad(self, batch, targets, weights):
        pred = self.sigmoid(batch.dot(weights))
        z = 2 * targets - 1
        return -np.sum(((1 - pred) * z).reshape(-1, 1) * batch, axis=0)

    def fit(self, X, y, num_epochs):
        X = self.basis_function(X)
        self.weights = self.optimizer.opt(X, y, self.weights_init(X.shape[1]), self.cost, self.cost_grad, num_epochs)

    def predict(self, X):
        assert self.weights is not None, "Model must be trained before predicting"
        return self.sigmoid(self.basis_function(X).dot(self.weights))
