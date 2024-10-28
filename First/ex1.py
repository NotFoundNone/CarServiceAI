import json

import numpy as np
from matplotlib import pyplot as plt

from First.computeCost import computeCost
from First.gradientDescent import gradientDescent, gradientDescentVector
from First.plotCostSurface import plotCostSurface
from First.plotData import plotDataGraphics

def loadData(file_name):
    X = []
    y = []
    with open(file_name, 'r') as file:
        for line in file:
            data = line.strip().split(',')
            X.append([1, float(data[0])])  # Добавляем 1 для theta_0
            y.append(float(data[1]))
    return X, y

def saveThetaToFile(theta, filename='theta.json'):
    with open(filename, 'w') as f:
        json.dump({'theta_0': theta[0], 'theta_1': theta[1]}, f)

def main():
    # Загрузка данных из файла
    X, y = loadData('../ex1data1.txt')

    plotDataGraphics(X, y)

    # Проверяем загрузку данных
    print(f"Загруженные данные: {X[:100]}, {y[:100]}")  # Выводим первые 100 строк для проверки

    # Инициализация параметров
    theta = [0, 0]  # Параметры theta
    alpha = 0.01    # Скорость обучения
    num_iters = 15000  # Количество итераций

    print(computeCost(X, y,theta))

    # Обучение модели с помощью градиентного спуска
    theta, theta_history, cost_history = gradientDescent(X, y, theta, alpha, num_iters)

    # Визуализируем итоговую прямую регрессии
    plt.figure(1)
    X_values = [x[1] for x in X]  # Извлекаем только значения переменной (второй элемент в каждом списке)
    plt.plot(X_values, [theta[0] + theta[1] * x for x in X_values], label='Linear regression')
    plt.scatter(X_values, y, marker='x', c='red', label='Training data')
    plt.legend()
    plt.title('Линейная регрессия')
    plt.xlabel('Количество автомобилей')
    plt.ylabel('Прибыль СТО')
    plt.grid(True)

    # Визуализируем историю изменения функции стоимости
    plt.figure(2)
    plt.plot(range(1, num_iters + 1), cost_history, label='Cost History')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost Function History')
    plt.grid(True)

    plotCostSurface(X, y, theta_history, cost_history)

    print(f"Обученные параметры: theta0 = {theta[0]}, theta1 = {theta[1]}")

    # Сохраняем параметры theta в файл
    saveThetaToFile(theta, filename="/Users/nikita/Desktop/CarSTO (1)/First/theta.json")

if __name__ == "__main__":
    main()