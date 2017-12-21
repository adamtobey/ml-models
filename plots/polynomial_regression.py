import numpy as np

from plotting import SingleClassPlot, ParametricPlotContainer, X_RANGE
from linear_basis_functions import ScalarBasisFunctions
from models.linear_regression import LinearRegression

class InteractivePolynomialRegression(SingleClassPlot):

    PLOT_POINTS = 150

    def __init__(self):
        self.plot = SingleClassPlot()
        self.container = ParametricPlotContainer(self.plot)

        self.container.slider(value=1, start=0, end=15, step=1, title="Polynomial Degree")
        self.container.slider(value=0, start=0, end=10, step=0.1, title="L2 Weight Penalty")

        self.fit_line = self.plot.figure.line(x=[], y=[])

        self.plot.enable_interaction()
        self.plot.add_change_listener(self.update_plot)

    def update_plot(self, plot_state):
        data = plot_state['inputs']
        X, y = np.array(data['x']), np.array(data['y'])
        regressor = LinearRegression(
            basis_function = ScalarBasisFunctions.Polynomial(plot_state['Polynomial Degree']),
            l2_cost = plot_state['L2 Weight Penalty']
        )
        regressor.fit(X, y)
        inputs = np.linspace(*X_RANGE, self.PLOT_POINTS)
        self.fit_line.data_source.data = dict(x=inputs, y=regressor.predict(inputs))

    def render(self, doc):
        doc.add_root(self.container.drawable())
