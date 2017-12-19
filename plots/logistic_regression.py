from plotting import MulticlassParametricPlot
import numpy as np
from models.logistic_regression import LogisticRegression
from linear_basis_functions import ScalarBasisFunctions
from bokeh.models.widgets import Slider

class InteractiveLogisticRegression(MulticlassParametricPlot):

    LINE_ENDPOINTS = np.array([-1, 1]) * 1000

    def __init__(self, data):
        super().__init__(data)

    def make_scatters(self, data):
        scatters = {}
        scatters['red'] = self.figure.scatter(color='#FF0000', **data[0])
        scatters['blue'] = self.figure.scatter(color='#0000FF', **data[1])
        return scatters

    def update(self, scatters):
        X = []
        y = []
        for c, scatter in enumerate(scatters.values()):
            data = scatter.data_source.data
            X.extend([
                [x, y] for x, y in zip(data['x'], data['y'])
            ])
            y.extend([
                c for _ in range(len(data['x']))
            ])
        X = np.array(X)
        y = np.array(y)

        classifier = LogisticRegression()
        classifier.fit(X, y, 100)

        b, w1, w2 = classifier.weights
        x_bound = self.LINE_ENDPOINTS
        y_bound = -(w1 * x_bound + b) / w2
        self.decision_boundary.data_source.data = dict(x=x_bound, y=y_bound)

    def initialize_figure(self, figure, scatters):
        self.decision_boundary = figure.line(x=[], y=[])
        self.update(scatters)

    def update_figure(self, figure, scatters, new, old):
        self.update(scatters)
