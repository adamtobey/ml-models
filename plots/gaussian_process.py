import numpy as np
from plotting import InteractivePlot

class InteractiveGaussianProcess(InteractivePlot):

    # TODO
    LINE_ENDPOINTS = np.arange(0, 10, 0.1)

    def __init__(self, data, regressor):
        self.regressor = regressor
        super().__init__(data)

    def update(self, X_train, y_train):
        X_predict = self.LINE_ENDPOINTS.reshape(-1, 1)
        X_train = X_train.reshape(-1, 1)
        self.regressor.fit(X_train, y_train)
        mean, cov = self.regressor.predictive_params(X_predict)

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

    def update_figure(self, figure, scatter, new, old):
        X, y = np.array(new['x']), np.array(new['y'])
        self.update(X, y)
