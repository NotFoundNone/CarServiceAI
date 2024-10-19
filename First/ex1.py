import json

from First.gradientDescent import gradientDescent
from First.plotCostSurface import plotCostSurface
from First.plotData import plotData

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
    X, y = loadData('./ex1data1.txt')

    plotData(X, y)

    # Проверяем загрузку данных
    print(f"Загруженные данные: {X[:100]}, {y[:100]}")  # Выводим первые 5 строк для проверки

    # Инициализация параметров
    theta = [10, 20]  # Параметры theta
    alpha = 0.01    # Скорость обучения
    num_iters = 1500  # Количество итераций

    # Обучение модели с помощью градиентного спуска
    theta, theta_history, cost_history = gradientDescent(X, y, theta, alpha, num_iters)

    plotCostSurface(X, y, theta_history, cost_history)

    print(f"Обученные параметры: theta0 = {theta[0]}, theta1 = {theta[1]}")

    # Сохраняем параметры theta в файл
    saveThetaToFile(theta, filename="./theta.json")

if __name__ == "__main__":
    main()
