import numpy as np

class Function(object):

    def eval(self, X, y, w):
        raise NotImplemented

    def gradient(self, X, y, w):
        raise NotImplemented

    def hessian(self, X, y, w):
        raise NotImplemented

class GradientDescentOptimizer(object):

    def __init__(self, learning_rate, num_epochs):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

    def opt(self, function, X, y, weights_init):
        weights = weights_init
        for epoch in range(self.num_epochs):
            grad = function.gradient(X, y, weights)
            weights -= grad * self.learning_rate
        return weights

class NewtonsMethod(object):

    def __init__(self, gradient_stop_magnitude=1e-2, max_updates=500):
        self.gradient_stop_magnitude = gradient_stop_magnitude
        self.max_updates = max_updates

    def opt(self, function, X, y, weights_init):
        weights = weights_init
        for i in range(self.max_updates):
            grad = function.gradient(X, y, weights)
            hess = function.hessian(X, y, weights)
            weights -= np.linalg.inv(hess).dot(grad)
            print("weights: ", weights)
            print("grad mag: ", np.sum(grad**2))
            if np.sum(grad**2) < self.gradient_stop_magnitude:
                return weights
