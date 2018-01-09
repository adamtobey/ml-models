import numpy as np

class Function(object):

    def eval(self, X, y, w):
        raise NotImplemented

    def gradient(self, X, y, w):
        raise NotImplemented

    def hessian(self, X, y, w):
        raise NotImplemented

class MomentumOptimizer(object):

    def __init__(self, learning_rate, num_epochs, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.num_epochs = num_epochs

    def opt(self, function, X, y, weights_init):
        weights = weights_init
        dw = 0
        for epoch in range(self.num_epochs):
            grad = function.gradient(X, y, weights)
            dw = self.momentum * dw - self.learning_rate * grad
            weights += dw
        return weights

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
        print(self.gradient_stop_magnitude)
        self.max_updates = max_updates

    def opt(self, function, X, y, weights_init):
        weights = weights_init
        for i in range(self.max_updates):
            print(i)
            grad = function.gradient(X, y, weights)
            hess = function.hessian(X, y, weights)
            print(grad, hess)
            weights -= np.linalg.inv(hess).dot(grad)
            print("grad mag: ", np.sum(grad**2)**0.5, "weights mag", np.sum(weights**2)**0.5)
            if np.sum(grad**2)**0.5 < self.gradient_stop_magnitude:
                return weights
        print("Max updates")
        return weights
