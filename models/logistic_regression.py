import numpy as np
from optimizers import SGDOptimizer
from design_matrices import DesignMatrixTransformer, Regularizers, BasisFunctions

class LogisticRegression(object):

    def __init__(self, optimizer=SGDOptimizer(learning_rate=0.1), basis_function=BasisFunctions.Affine()):
        self.optimizer = optimizer
        self.weights = None
        self.weights_init = lambda s: np.random.rand(*s) * 0.1
        # TODO this regularization only works with least squares
        self.transformer = DesignMatrixTransformer(basis_function=basis_function, regularizer=regularizer)

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

    def fit(self, data_provider, num_epochs):
        print(num_epochs)
        data = self.transformer.train(data_provider)
        shape = self.transformer.shape(data_provider)
        self.weights = self.optimizer.opt(data, self.weights_init(shape), self.cost, self.cost_grad, num_epochs)

    def predict(self, inputs):
        assert self.weights is not None, "Model must be trained before predicting"
        return self.sigmoid(self.transformer.make_design_matrix(inputs).dot(self.weights))
