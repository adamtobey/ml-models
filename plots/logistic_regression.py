import numpy as np

from bokeh.models.widgets import Slider

from plotting import MultiClassPlot, X_RANGE
from linear_basis_functions import ScalarBasisFunctions
from models.logistic_regression import LogisticRegression

class InteractiveLogisticRegression(object):

    def __init__(self):
        self.plot = MultiClassPlot()

        self.plot.add_scatter('red', color='#FF0000', x=[], y=[])
        self.plot.add_scatter('blue', color='#0000FF', x=[], y=[])

        self.decision_boundary = self.plot.figure.line(x=[], y=[])

        self.plot.add_change_listener(self.update_plot)
        self.plot.enable_interaction()

    def plot_state_to_model_data(self, plot_state):
        X = []
        y = []
        for clas, data in enumerate(plot_state['inputs'].values()):
            X.extend([
                [x, y] for x, y in zip(data['x'], data['y'])
            ])
            y.extend([
                clas for _ in range(len(data['x']))
            ])
        X = np.array(X)
        y = np.array(y)
        return X, y

    def update_plot(self, plot_state):
        X, y = self.plot_state_to_model_data(plot_state)

        if X.shape[0] != 0:
            classifier = LogisticRegression()
            classifier.fit(X, y, 100)

            b, w1, w2 = classifier.weights
            x_bound = np.array([*X_RANGE])
            y_bound = -(w1 * x_bound + b) / w2
            self.decision_boundary.data_source.data = dict(x=x_bound, y=y_bound)

    def render(self, doc):
        doc.add_root(self.plot.drawable())
