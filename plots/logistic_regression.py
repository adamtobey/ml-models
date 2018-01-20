import math
import numpy as np

from plotting import MultiClassPlot, X_RANGE, Y_RANGE
from linear_basis_functions import ScalarBasisFunctions
from models.logistic_regression import LogisticRegression

def plot_state_to_model_data(plot_state):
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

def angle_from_tangent(t1, t2):
    h = (t1**2 + t2**2)**0.5
    offset = 0
    if t2 < 0:
        t1, t2 = -t1, -t2
        offset = np.pi
    return math.acos(t1 / h) + offset

def calculate_line_endpoints(angle, origin, x_range=X_RANGE, y_range=Y_RANGE):
    angle = angle % np.pi
    l1, l2 = math.cos(angle), math.sin(angle)
    x0, y0 = origin
    b1, b2 = x_range[1], y_range[1]
    c1, c2 = x_range[0], y_range[0]
    if l2 * (b1 - x0) > l1 * (b2 - y0):
        t = (np.array([b1, c1]) - y0) / l2
    else:
        t = (np.array([b2, c2]) - x0) / l1
    return l1 * t + x0, l2 * t + y0

class InteractiveLogisticRegression(object):

    CONFIDENCE_INTERVAL = 0.95
    ALPHA = -math.log(1 / CONFIDENCE_INTERVAL - 1, 2)

    def __init__(self):
        self.plot = MultiClassPlot()

        self.plot.add_scatter('red', color='#FF0000', x=[], y=[])
        self.plot.add_scatter('blue', color='#0000FF', x=[], y=[])

        self.decision_boundary = self.plot.figure.line(x=[], y=[])
        self.red_side = self.plot.figure.line(x=[], y=[], color="#FF0000", line_dash="dashed")
        self.blue_side = self.plot.figure.line(x=[], y=[], color="#0000FF", line_dash="dashed")

        self.plot.add_change_listener(self.update_plot)
        self.plot.enable_interaction()

    def update_plot(self, plot_state):
        X, y = plot_state_to_model_data(plot_state)

        if X.shape[0] != 0:
            classifier = LogisticRegression()
            classifier.fit(X, y, 1000)

            b, w1, w2 = classifier.weights

            origin = np.array([0, -b / w2])
            angle = angle_from_tangent(-w2, w1)

            delta = np.array([0, self.ALPHA / w2])

            x_red, y_red = calculate_line_endpoints(angle, origin - delta)
            x_decision, y_decision = calculate_line_endpoints(angle, origin)
            x_blue, y_blue = calculate_line_endpoints(angle, origin + delta)

            self.decision_boundary.data_source.data = dict(x=x_decision, y=y_decision)
            self.red_side.data_source.data = dict(x=x_red, y=y_red)
            self.blue_side.data_source.data = dict(x=x_blue, y=y_blue)

    def render(self, doc):
        doc.add_root(self.plot.drawable())
