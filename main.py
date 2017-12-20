import os
import logging
from threading import Thread
from flask import Flask, render_template

from bokeh.embed import server_document
from bokeh.server.server import Server

from plots import InteractivePolynomialRegression, InteractiveGaussianProcess, InteractiveBayesianPolynomialRegression, InteractiveLogisticRegression

app = Flask(__name__)


HTTP_PORT = int(os.environ.get("PORT", 8000))
BOKEH_PORT = int(os.environ.get("BOKEH_PORT", 5006))
HOSTNAME = os.environ.get("HOSTNAME", "localhost")


logging.warning("HTTP_PORT: {}, BOKEH_PORT: {}, HOSTNAME: {}".format(HTTP_PORT, BOKEH_PORT, HOSTNAME))


plot_pages = dict(
    linear_regression=("Polynomial Regression", InteractivePolynomialRegression),
    gaussian_process=("Gaussian Process Regression", InteractiveGaussianProcess),
    bayesian_linear_regression=("Bayesian Polynomial Regression", InteractiveBayesianPolynomialRegression),
    logistic_regression=("Logistic Regression", InteractiveLogisticRegression)
)

bokeh_routes = {}

def make_bokeh_route(Plot):
    def make_plot(doc):
        Plot().render(doc)
    return make_plot

def make_flask_route(route, name):
    @app.route("/{}".format(route), methods=["GET"], endpoint=route)
    def make_page():
        script = server_document("http://localhost:5006/plots/{}".format(route))
        return render_template("plot.html", model_name=name, routes=plot_pages, script=script, template="Flask")

for route, (name, Plot) in plot_pages.items():
    bokeh_routes["/plots/{}".format(route)] = make_bokeh_route(Plot)
    make_flask_route(route, name)

def bokeh_worker(routes):
    def worker():
        server = Server(routes, allow_websocket_origin=["{}:{}".format(HOSTNAME, HTTP_PORT)], port=BOKEH_PORT)
        server.start()
        server.io_loop.start()
    return worker

Thread(target = bokeh_worker(bokeh_routes)).start()

if __name__ == '__main__':
    app.run(port=HTTP_PORT)
