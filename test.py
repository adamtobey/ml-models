from models.linear_regression import LinearRegression, ScalarBasisFunctions
import numpy as np

regressor = LinearRegression(basis_function=ScalarBasisFunctions.Polynomial(2))

x_train = np.arange(4)
y_train = x_train ** 2 + x_train + 3

x_test = np.arange(4, 6)
y_test = x_test ** 2 + x_test + 3

regressor.fit(x_train, y_train)

print("Model fit: ", regressor.predict(x_test))
print("Y test: ", y_test)
