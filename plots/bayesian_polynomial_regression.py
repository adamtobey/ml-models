from plotting import InteractiveParametricPlot
import numpy as np
from models.bayesian_linear_regression import BayesianLinearRegression
from linear_basis_functions import ScalarBasisFunctions
from bokeh.models.widgets import Slider

class InteractiveBayesianPolynomialRegression(InteractiveParametricPlot):

    # TODO
    LINE_ENDPOINTS = np.arange(0, 10, 0.1)

    def parameters(self):
        params = {}
        params['degree'] = Slider(value=1, start=0, end=15, step=1, title="Polynomial Degree")
        params['prior_variance'] = Slider(value=0.4, start=0.01, end=2, step=0.01, title="Prior Variance")
        params['s2y'] = Slider(value=0.1, start=0.01, end=2, step=0.01, title="Noise Variance")
        return params

    def update(self, X_train, y_train):
        degree, prior_variance, s2y = [self.params[k].value for k in ['degree', 'prior_variance', 's2y']]

        basis = ScalarBasisFunctions.Polynomial(degree)
        X_predict = basis(self.LINE_ENDPOINTS)

        regressor = BayesianLinearRegression(prior_mean=np.zeros((degree + 1,)), prior_covariance=prior_variance*np.identity(degree + 1), s2y=s2y)

        regressor.fit(basis(X_train), y_train)
        means, vars = regressor.predictive_params(X_predict)

        # Point estimate
        x_fit, y_fit = self.LINE_ENDPOINTS, means
        self.point_estimate.data_source.data = dict(x=x_fit, y=y_fit)

        # Uncertainty range
        x_range, y_range = [], []
        x_range.extend(self.LINE_ENDPOINTS)
        y_range.extend(means + vars)
        x_range.extend(reversed(self.LINE_ENDPOINTS))
        y_range.extend(reversed(means - vars))
        self.uncertainty_range.data_source.data = dict(x=x_range, y=y_range)

    def initialize_figure(self, figure, scatter):
        data = scatter.data_source.data
        X, y = np.array(data['x']), np.array(data['y'])
        self.point_estimate = figure.line(x=[], y=[])
        self.uncertainty_range = figure.patch(x=[], y=[], fill_alpha=0.5)
        self.update(X, y)

    def update_figure(self, figure, scatter, new, old):
        data = scatter.data_source.data
        X, y = np.array(data['x']), np.array(data['y'])
        self.update(X, y)
