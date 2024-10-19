
#Функция стоимости
def computeCost(X, y, theta):
    m = len(y)
    total_cost = 0
    for i in range(m):
        hypothesis = X[i][0] * theta[0] + X[i][1] * theta[1]
        total_cost += (hypothesis - y[i]) ** 2
    cost = total_cost / (2 * m)
    return cost
