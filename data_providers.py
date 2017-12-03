class BasicDataProvider(object):

    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
        self.shape = inputs.shape[1:]

    def train(self):
        yield self.inputs, self.targets
