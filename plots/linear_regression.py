from plotting import InteractivePlot
import numpy as np

class InteractiveLinearRegression(InteractivePlot):

    LINE_ENDPOINTS = np.array([-1, 1]) * 1000

    def __init__(self, data, regressor):
        self.regressor = regressor
        super().__init__(data)

    def initialize_figure(self, figure, scatter):
        data = scatter.data_source.data
        X, y = np.array(data['x']), np.array(data['y'])
        self.regressor.fit(X, y)
        x_fit, y_fit = self.LINE_ENDPOINTS, self.regressor.predict(self.LINE_ENDPOINTS)
        self.fit_line = figure.line(x=x_fit, y=y_fit)

    def update_figure(self, figure, scatter, new, old):
        X, y = np.array(new['x']), np.array(new['y'])
        self.regressor.fit(X, y)
        self.fit_line.data_source.data = dict(x=self.LINE_ENDPOINTS, y=self.regressor.predict(self.LINE_ENDPOINTS))
