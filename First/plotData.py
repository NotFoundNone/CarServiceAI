from matplotlib import pyplot as plt


#Функция, визуализирующая обучающие данные
def plotData(x, y):
    # Простой вывод данных в виде пар значений
    print("Количество автомобилей и прибыль:")
    for i in range(len(x)):
        print(f"Автомобили: {x[i]}, Прибыль: {y[i]}")

def plotDataGraphics(X, y):
    # Извлекаем только значения переменной (второй элемент в каждом списке)
    X_values = [x[1] for x in X]

    plt.scatter(X_values, y, marker='x', c='red', label='Обучающая выборка')
    plt.xlabel('Количество авто')
    plt.ylabel('Прибыль')
    plt.title('Прибыль в зависимости от количества автомобилей')
    plt.grid(True)
    plt.legend()
    plt.show()
