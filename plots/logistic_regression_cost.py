import math
import numpy as np

from bokeh.plotting import figure
from bokeh.layouts import column

from plotting import MultiClassPlot, ParametricPlotContainer, X_RANGE, Y_RANGE, BaseInteractivePlot
from models.logistic_regression import LogisticRegression
from plots.logistic_regression import plot_state_to_model_data, calculate_line_endpoints
from linear_basis_functions import BasisFunctions

class LogisticRegressionCostPlot(object):

    EXPLORE_PLOT_KEYS = ["Line x Position", "Line Angle", "Confidence Interval"]
    PLOT_POINTS = 75
    CONFIDENCE_INTERVAL = 0.95
    ALPHA = -math.log(1 / CONFIDENCE_INTERVAL - 1, 2)

    basis = staticmethod(BasisFunctions.Affine())

    def __init__(self):
        self.plot = MultiClassPlot()
        self.main_container = ParametricPlotContainer(self.plot)

        self.plot.add_scatter("red", x=[], y=[], color="#FF0000")
        self.plot.add_scatter("blue", x=[], y=[], color="#0000FF")

        self.main_container.slider(title="Line x Position", value=sum(X_RANGE) / 2, start=X_RANGE[0], end=X_RANGE[1], step=0.1)
        self.main_container.slider(title="Line Angle", value=0, start=0, end=360, step=1)
        self.main_container.slider(title="Confidence Interval", value=1, start=0, end=20, step=0.1)

        COST_RANGE = (0, 20)
        self.explore_figures = {
            "Line x Position": figure(x_range=X_RANGE, y_range=COST_RANGE, tools="", toolbar_location=None),
            "Line Angle": figure(x_range=(0, 360), y_range=COST_RANGE, tools="", toolbar_location=None),
            "Confidence Interval": figure(x_range=(0, 20), y_range=COST_RANGE, tools="", toolbar_location=None)
        }
        self.explore_plots = {
            key: (
                fig.line(x=[], y=[]),
                fig.line(x=[], y=[])
            )
            for key, fig in self.explore_figures.items()
        }
        self.decision_boundary = self.plot.figure.line(x=[], y=[])
        self.red_side = self.plot.figure.line(x=[], y=[], color="#FF0000", line_dash="dashed")
        self.blue_side = self.plot.figure.line(x=[], y=[], color="#0000FF", line_dash="dashed")

        self.cache = {
            "Line x Position": None,
            "Line Angle": None,
            "Confidence Interval": None
        }

        self.plot.enable_interaction()
        self.plot.add_change_listener(self.update_plot)

    def deg_to_rad(self, x):
        return np.pi * x / 180

    def explore_weights_over_range(self, x_range, inputs):
        out = np.ndarray((len(inputs), x_range.shape[0]))
        w1 = out[1,:] = np.array(self.ALPHA / inputs["Confidence Interval"])
        out[0,:] = np.array(-w1 * inputs["Line x Position"])
        out[2,:] = np.array(-w1 / np.sin(self.deg_to_rad(inputs["Line Angle"])))
        return out

    def cost(self, X, y, W):
        affine = X.dot(W)
        z = 2 * y - 1
        return -np.sum(np.log(1 / (1 + np.exp(z * affine))), axis=0)

    def update_explore_plot(self, key, plot_state):
        X, y = plot_state_to_model_data(plot_state)
        X = self.basis(X)
        inputs = {
            const: plot_state[const] for const in self.EXPLORE_PLOT_KEYS if const != key
        }
        plot = self.explore_plots[key][0]
        x_range = self.explore_figures[key].x_range
        x_plot = np.linspace(x_range.start, x_range.end, self.PLOT_POINTS)
        inputs[key] = x_plot
        W = self.explore_weights_over_range(x_plot, inputs)
        y_plot = self.cost(X, y, W)
        plot.data_source.data = dict(x=x_plot, y=y_plot)

    def update_plot(self, plot_state):
        for key in self.EXPLORE_PLOT_KEYS:
            self.explore_plots[key][1].data_source.data = dict(
                x=[plot_state[key], plot_state[key]],
                y=[*Y_RANGE]
            )
            if self.cache[key] == plot_state[key]:
                self.update_explore_plot(key, plot_state)
        self.cache = {
            key: plot_state[key] for key in self.EXPLORE_PLOT_KEYS
        }

        # w2 = math.sqrt(self.ALPHA / (plot_state["Confidence Interval"] * (1 + math.tan(theta))))

        x0 = plot_state["Line x Position"]
        decision_origin = np.array([x0, 0])
        theta = self.deg_to_rad(plot_state["Line Angle"])
        offset = np.array([-math.sin(theta), math.cos(theta)]) * plot_state["Confidence Interval"]

        for plot, origin in [
            (self.blue_side, decision_origin - offset),
            (self.decision_boundary, decision_origin),
            (self.red_side, decision_origin + offset)
        ]:
            x, y = calculate_line_endpoints(plot_state["Line Angle"], origin)
            plot.data_source.data = dict(x=x, y=y)

    def render(self, doc):
        doc.add_root(column(
            self.main_container.drawable(),
            *self.explore_figures.values()
        ))
