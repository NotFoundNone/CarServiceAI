import numpy as np

vector1 = np.array([[1], [2], [3]])
vector2 = np.array([[1], [2], [3]])

print("Формат vector1", vector1.shape)
print("Формат vector2:", vector2.shape)

scalar_product_loops = 0
for i in range(vector1.shape[0]):
    scalar_product_loops += vector1[i][0] * vector2[i][0]
print("Скалярное произведение (циклы):", scalar_product_loops)

scalar_product_elementwise = np.sum(vector1 * vector2)
print("Произведение поэлементно:", scalar_product_elementwise)

scalar_product_matrix = np.dot(vector1.T, vector2)
print("Произведение матричным методом:", scalar_product_matrix[0][0])

print(scalar_product_matrix.shape)