import numpy as np

from plotting import SingleClassPlot, ParametricPlotContainer, X_RANGE
from models.gaussian_process import GaussianProcessRegression, Kernels

class InteractiveGaussianProcess(SingleClassPlot):

    PLOT_POINTS = 150

    def __init__(self):
        self.plot = SingleClassPlot()
        self.container = ParametricPlotContainer(self.plot)

        self.container.slider(value=0.5, start=0.1, end=5, step=0.1, title="Kernel Width")
        self.container.slider(value=1, start=0.1, end=5, step=0.1, title="Kernel Amplitude")
        self.container.slider(value=0.1, start=0.01, end=1, step=0.01, title="Prior Variance")

        self.point_estimate = self.plot.figure.line(x=[], y=[])
        self.uncertainty_range = self.plot.figure.patch(x=[], y=[], fill_alpha=0.5)

        self.plot.add_change_listener(self.update_plot)
        self.plot.enable_interaction()

    def update_plot(self, plot_state):
        inputs = np.linspace(*X_RANGE, self.PLOT_POINTS)
        X_predict = inputs.reshape(-1, 1)
        data = plot_state['inputs']
        X_train = np.array(data['x']).reshape(-1, 1)
        y_train = np.array(data['y'])

        regressor = GaussianProcessRegression(
            kernel=Kernels.GaussianKernel(
                plot_state['Kernel Amplitude'],
                np.array([plot_state['Kernel Width']])),
            s2y=plot_state['Prior Variance'])
        regressor.fit(X_train, y_train)
        mean, cov = regressor.predictive_params(X_predict)

        # Point estimate
        x_fit, y_fit = inputs, mean
        self.point_estimate.data_source.data = dict(x=x_fit, y=y_fit)

        # Uncertainty range
        x_range, y_range = [], []
        x_range.extend(inputs)
        y_range.extend(mean + cov)
        x_range.extend(reversed(inputs))
        y_range.extend(reversed(mean - cov))
        self.uncertainty_range.data_source.data = dict(x=x_range, y=y_range)

    def render(self, doc):
        doc.add_root(self.container.drawable())
