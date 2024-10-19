
def warmUpExercise_with_standard_functions(n):
    #Создание единичной матрицы с использованием стандартных функций
    I = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    return I

def warmUpExercise_without_standard_functions(n):
    #Создание единичной матрицы без использования стандартных функций.
    I = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(1)
            else:
                row.append(0)
        I.append(row)
    return I
