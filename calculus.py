import numpy as np

def gradient(fun, point, delta=1e-3):
    D = delta * np.identity(point.shape[0])
    return np.array(list(map(lambda p: fun(point + p) - fun(point - p), D))) / 2 / delta

def hessian(fun, point, delta=1e-3):
    return gradient(lambda x: gradient(fun, x, delta / 2), point, delta)

def cost(x, t, w):
    z = 2 * t - 1
    a = np.exp(-z * w.dot(x))
    return np.log(1 + a)

def sigmoid_hessian(x, t, w):
    z = 2 * t - 1
    a = np.exp(-z * w.dot(x))
    diag = np.ones(x.shape) * z * (a**2 + a) / (1 + a)**2
    return np.outer(x, x) * a / (1 + a)**2 + np.diag(diag)

def f(x):
    return x.dot(x)

# print("Gradient: ", gradient(f, np.arange(4)))
# print("Hessian: ", hessian(lambda x: sum(x) ** 2, np.arange(4)))

w = np.arange(4)
x = w + 5
for t in [0, 1]:
#     print(f"x: {x}, t: {t}")
#     print("Numerical: ", hessian(lambda x: cost(x, t, w), x))
    print("Analytic: ", sigmoid_hessian(x, t, w))
