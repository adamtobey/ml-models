class SGDOptimizer(object):

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def opt(self, batches, weights_init, cost, cost_grad, num_epochs):
        weights = weights_init
        for epoch in range(num_epochs):
            for batch, targets in batches():
                grad = cost_grad(batch, targets, weights)
                weights -= grad * self.learning_rate
        return weights
