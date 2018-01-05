import numpy as np
from calculus import gradient
from models.logistic_regression import LogisticRegressionCost, RefLRC

x = np.arange(4).reshape((2,2))
w = y = np.arange(2)

r = RefLRC()
t = LogisticRegressionCost()

ag = gradient(lambda x: t.eval(x, y, w), x)
ah = gradient(lambda x: t.gradient(x, y, w), x)

print("Anal gradient: ", ag)
print("Ref gradient: ", r.gradient(x, y, w))
print("Test gradient: ", t.gradient(x, y, w))
print("Anal hessian: ", ah)
print("Ref hessian: ", r.hessian(x, y, w))
print("Test hessian: ", t.hessian(x, y, w))
