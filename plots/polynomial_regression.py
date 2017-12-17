from plotting import InteractiveParametricPlot
from models.linear_regression import LinearRegression
from linear_basis_functions import ScalarBasisFunctions
from bokeh.models.widgets import Slider
import numpy as np

class InteractivePolynomialRegression(InteractiveParametricPlot):

    LINE_ENDPOINTS = np.arange(0, 10, 0.1)

    def __init__(self, data):
        super().__init__(data)

    def parameters(self):
        params = {}
        params['degree'] = Slider(value=1, start=0, end=15, step=1, title="Polynomial Degree")
        params['l2_cost'] = Slider(value=0, start=0, end=10, step=0.1, title="L2 Weight Penalty")
        return params

    def fit(self, scatter):
        data = scatter.data_source.data
        X, y = np.array(data['x']), np.array(data['y'])
        degree, l2_cost = [self.params[k].value for k in ['degree', 'l2_cost']]
        regressor = LinearRegression(basis_function=ScalarBasisFunctions.Polynomial(degree), l2_cost=l2_cost)
        regressor.fit(X, y)
        return self.LINE_ENDPOINTS, regressor.predict(self.LINE_ENDPOINTS)

    def initialize_figure(self, figure, scatter):
        x, y = self.fit(scatter)
        self.fit_line = figure.line(x=x, y=y)

    def update_figure(self, figure, scatter, a, b):
        x, y = self.fit(scatter)
        self.fit_line.data_source.data = dict(x=x, y=y)
