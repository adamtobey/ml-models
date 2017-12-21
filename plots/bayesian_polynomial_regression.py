import numpy as np

from plotting import SingleClassPlot, ParametricPlotContainer, X_RANGE
from linear_basis_functions import ScalarBasisFunctions
from models.bayesian_linear_regression import BayesianLinearRegression

class InteractiveBayesianPolynomialRegression(SingleClassPlot):

    PLOT_POINTS = 150

    def __init__(self):
        self.plot = SingleClassPlot()
        self.container = ParametricPlotContainer(self.plot)

        self.container.slider(value=1, start=0, end=15, step=1, title="Polynomial Degree")
        self.container.slider(value=0.4, start=0.01, end=5, step=0.01, title="Prior Variance")
        self.container.slider(value=0.1, start=0.01, end=5, step=0.01, title="Noise Variance")

        self.point_estimate = self.plot.figure.line(x=[], y=[])
        self.uncertainty_range = self.plot.figure.patch(x=[], y=[], fill_alpha=0.5)

        self.plot.enable_interaction()
        self.plot.add_change_listener(self.update_plot)

    def update_plot(self, plot_state):
        inputs = np.linspace(*X_RANGE, self.PLOT_POINTS)
        basis = ScalarBasisFunctions.Polynomial(plot_state['Polynomial Degree'])
        X_predict = basis(inputs)

        regressor = BayesianLinearRegression(
            prior_mean = np.zeros((plot_state['Polynomial Degree'] + 1,)),
            prior_covariance = plot_state['Prior Variance'] * np.identity(plot_state['Polynomial Degree'] + 1),
            s2y = plot_state['Noise Variance']
        )

        data = plot_state['inputs']
        regressor.fit(basis(np.array(data['x'])), np.array(data['y']))
        means, vars = regressor.predictive_params(X_predict)

        # Point estimate
        x_fit, y_fit = inputs, means
        self.point_estimate.data_source.data = dict(x=x_fit, y=y_fit)

        # Uncertainty range
        x_range, y_range = [], []
        x_range.extend(inputs)
        y_range.extend(means + vars)
        x_range.extend(reversed(inputs))
        y_range.extend(reversed(means - vars))
        self.uncertainty_range.data_source.data = dict(x=x_range, y=y_range)

    def render(self, doc):
        doc.add_root(self.container.drawable())
