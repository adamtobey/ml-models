class GradientDescentOptimizer(object):

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def opt(self, X, y, weights_init, cost, cost_grad, num_epochs):
        weights = weights_init
        for epoch in range(num_epochs):
            grad = cost_grad(X, y, weights)
            weights -= grad * self.learning_rate
        return weights
