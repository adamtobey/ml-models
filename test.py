from models.linear_regression import LinearRegression
import numpy as np

regressor = LinearRegression()

x_train = np.array([[1, 0], [1, 1], [1, 2]])
y_train = np.array([3, 5, 7])

x_test = np.array([[1, 5], [1, 6]])
y_test = np.array([13, 15])

regressor.fit(x_train, y_train)

print("Model fit: ", regressor.predict(x_test))
print("Y test: ", y_test)
