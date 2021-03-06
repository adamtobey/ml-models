import os

from jinja2 import Environment, FileSystemLoader
from tornado.web import RequestHandler

from bokeh.embed import server_document
from bokeh.server.server import Server

from plots import InteractivePolynomialRegression, InteractiveGaussianProcess, InteractiveBayesianPolynomialRegression, InteractiveLogisticRegression, LogisticRegressionCostPlot, LogisticRegressionCost1D


HTTP_PORT = int(os.environ.get("PORT", 8000))
HOSTNAME = os.environ.get("HOSTNAME", "localhost:{}".format(HTTP_PORT))
HOST_PROTOCOL = os.environ.get("HOST_PROTOCOL", "http")
HOST_URL = "{}://{}".format(HOST_PROTOCOL, HOSTNAME)

env = Environment(loader=FileSystemLoader("templates"))

plot_pages = dict(
    linear_regression=("Polynomial", InteractivePolynomialRegression),
    gaussian_process=("Gaussian Process", InteractiveGaussianProcess),
    bayesian_linear_regression=("Bayesian Polynomial", InteractiveBayesianPolynomialRegression),
    logistic_regression=("Logistic", InteractiveLogisticRegression),
    logistic_regression_cost_1_d=("Logistic Cost", LogisticRegressionCost1D)
)

bokeh_routes = {}
page_routes = {}

def bokeh_route(route):
    return "/plots/{}".format(route)

def page_route(route):
    return "/{}".format(route)

def make_bokeh_route(Plot):
    def make_plot(doc):
        Plot().render(doc)
    return make_plot

def make_page_route(name, route):
    global env, HOST_URL, plot_pages
    class PageRoute(RequestHandler):
        def get(self):
            template = env.get_template("plot.html")
            script = server_document(HOST_URL + bokeh_route(route))
            self.write(template.render(script=script, model_name=name, routes=plot_pages))
    return PageRoute

for route, (name, Plot) in plot_pages.items():
    bokeh_routes[bokeh_route(route)] = make_bokeh_route(Plot)
    page_routes[page_route(route)] = make_page_route(name, route)

class AboutPage(RequestHandler):
    def get(self):
        template = env.get_template("about.html")
        self.write(template.render(routes=plot_pages))

page_routes["/"] = AboutPage

server = Server(
    bokeh_routes,
    extra_patterns=list(page_routes.items()),
    port=HTTP_PORT,
    allow_websocket_origin=[HOSTNAME]
)
server.start()

if __name__ == '__main__':
    print("Starting the server on port {}".format(HTTP_PORT))
    server.io_loop.start()
