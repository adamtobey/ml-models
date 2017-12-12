from plotting import InteractivePlot
from data_providers import BasicDataProvider
import numpy as np

class InteractiveLinearRegression(InteractivePlot):

    LINE_ENDPOINTS = np.array([-1, 1]) * 1000

    def __init__(self, regressor):
        self.regressor = regressor
        super().__init__()

    def initialize_figure(self, figure, scatter):
        data = scatter.data_source.data
        X, y = np.array(data['x']), np.array(data['y'])
        self.regressor.fit(dp)
        x_fit, y_fit = LINE_ENDPOINTS, self.regressor.predict(LINE_ENDPOINTS)
        self.fit_line = p.line(x=x_fit, y=y_fit)

    def update_figure(self, figure, scatter, new, old):
        X, y = np.array(new['x']), np.array(new['y'])
        self.regressor.fit(dp)
        self.fit_line.data_source.data = dict(x=LINE_ENDPOINTS, y=self.regressor.predict(LINE_ENDPOINTS))
