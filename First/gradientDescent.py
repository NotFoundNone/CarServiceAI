import numpy as np

from First.computeCost import computeCost, computeCostVector

#Градиент
def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    cost_history = []
    theta_history = []

    for _ in range(num_iters):
        sum_theta0 = 0
        sum_theta1 = 0
        for i in range(m):
            hypothesis = theta[0] * X[i][0] + theta[1] * X[i][1]
            error = hypothesis - y[i]
            sum_theta0 += error * X[i][0]
            sum_theta1 += error * X[i][1]
        theta[0] -= (alpha / m) * sum_theta0
        theta[1] -= (alpha / m) * sum_theta1

        theta_history.append(theta.copy())
        cost_history.append(computeCost(X, y, theta))
    return theta, theta_history, cost_history

def gradientDescentVector(X, y, theta, alpha, iterations):
    X = np.array(X)
    y = np.array(y)
    theta = np.array(theta, dtype=float)

    m = len(y)
    cost_history = []
    theta_history = []

    for _ in range(iterations):
        # Вычисляем предсказания для всех примеров одновременно (векторизованно)
        predictions = X.dot(theta)

        # Вычисляем разницу между предсказаниями и реальными значениями (вектор ошибок)
        errors = predictions - y

        # Вычисляем градиент (векторизованно)
        gradient = (1 / m) * X.T.dot(errors)

        # Обновляем параметры theta (векторизованно)
        theta -= alpha * gradient

        # Сохраняем историю значений theta и стоимости
        theta_history.append(theta.copy())
        cost_history.append(computeCostVector(X, y, theta))

    return theta, theta_history, cost_history