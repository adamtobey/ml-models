import os

from jinja2 import Environment, FileSystemLoader
from tornado.web import RequestHandler

from bokeh.embed import server_document
from bokeh.server.server import Server

from plots import InteractivePolynomialRegression, InteractiveGaussianProcess, InteractiveBayesianPolynomialRegression, InteractiveLogisticRegression


HTTP_PORT = int(os.environ.get("PORT", 8000))
HOST_URL = os.environ.get("HOST_URL", "http://localhost:{}".format(HTTP_PORT))

env = Environment(loader=FileSystemLoader("templates"))

plot_pages = dict(
    linear_regression=("Polynomial Regression", InteractivePolynomialRegression),
    gaussian_process=("Gaussian Process Regression", InteractiveGaussianProcess),
    bayesian_linear_regression=("Bayesian Polynomial Regression", InteractiveBayesianPolynomialRegression),
    logistic_regression=("Logistic Regression", InteractiveLogisticRegression)
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

server = Server(
    bokeh_routes,
    extra_patterns=list(page_routes.items()),
    port=HTTP_PORT,
    allow_websocket_origin=["calm-cliffs-70784.herokuapp.com"]
)
server.start()

if __name__ == '__main__':
    print("Starting the server on port {}".format(HTTP_PORT))
    server.io_loop.start()
