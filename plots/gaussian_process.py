import numpy as np
from plotting import InteractiveParametricPlot
from bokeh.models.widgets import Slider
from models.gaussian_process import GaussianProcessRegression, Kernels

class InteractiveGaussianProcess(InteractiveParametricPlot):

    # TODO
    LINE_ENDPOINTS = np.arange(0, 10, 0.1)

    def parameters(self):
        params = {}
        params['l'] = Slider(value=0.5, start=0.1, end=5, step=0.1, title="Kernel Width")
        params['s2'] = Slider(value=1, start=0.1, end=5, step=0.1, title="Kernel Amplitude")
        params['s2y'] = Slider(value=0.1, start=0.01, end=1, step=0.01, title="Prior Variance")
        return params

    def update(self, X_train, y_train):
        X_predict = self.LINE_ENDPOINTS.reshape(-1, 1)
        X_train = X_train.reshape(-1, 1)

        l, s2, s2y = [self.params[k].value for k in ['l', 's2', 's2y']]
        regressor = GaussianProcessRegression(kernel=Kernels.GaussianKernel(s2, np.array([l])), s2y=s2y)
        regressor.fit(X_train, y_train)
        mean, cov = regressor.predictive_params(X_predict)

        # Point estimate
        x_fit, y_fit = self.LINE_ENDPOINTS, mean
        self.point_estimate.data_source.data = dict(x=x_fit, y=y_fit)

        # Uncertainty range
        x_range, y_range = [], []
        x_range.extend(self.LINE_ENDPOINTS)
        y_range.extend(mean + cov)
        x_range.extend(reversed(self.LINE_ENDPOINTS))
        y_range.extend(reversed(mean - cov))
        self.uncertainty_range.data_source.data = dict(x=x_range, y=y_range)

    def initialize_figure(self, figure, scatter):
        data = scatter.data_source.data
        X, y = np.array(data['x']), np.array(data['y'])
        self.point_estimate = figure.line(x=[], y=[])
        self.uncertainty_range = figure.patch(x=[], y=[], fill_alpha=0.5)
        self.update(X, y)

    def update_figure(self, figure, scatter, a, b):
        data = scatter.data_source.data
        X, y = np.array(data['x']), np.array(data['y'])
        self.update(X, y)
