import numpy as np

class SGDOptimizer(object):

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def opt(self, data_provider, weights_init, cost, cost_grad, num_epochs):
        weights = weights_init
        for epoch in range(num_epochs):
            for batch, targets in data_provider.train():
                grad = cost_grad(batch, targets, weights)
                weights -= grad * self.learning_rate
        return weights

class BasicDataProvider(object):

    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
        self.shape = inputs.shape[1:]

    def train(self):
        yield self.inputs, self.targets


class LogisticRegression(object):

    def __init__(self, optimizer=SGDOptimizer(learning_rate=0.1)):
        self.optimizer = optimizer
        self.weights = None
        self.weights_init = lambda s: np.random.rand(*s) * 0.1

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
        self.weights = self.optimizer.opt(data_provider, self.weights_init(data_provider.shape), self.cost, self.cost_grad, num_epochs)

    def predict(self, inputs):
        assert self.weights is not None, "Model must be trained before predicting"
        return self.sigmoid(inputs.dot(self.weights))
