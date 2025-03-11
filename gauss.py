import numpy as np
import copy
from utils import print_matrix


def gaussian_elimination_pure(A, b):
    n = len(A)

    # Создаем копии, чтобы сохранить исходные данные
    A_orig = copy.deepcopy(A)
    b_orig = b.copy()

    A = [row[:] for row in A]
    b = b[:]

    # Прямой ход (приведение к верхнетреугольному виду)
    for i in range(n):
        max_row = i
        for k in range(i + 1, n):
            if abs(A[k][i]) > abs(A[max_row][i]):
                max_row = k

        if A[max_row][i] == 0:
            raise ValueError("Матрица вырожденная, решение невозможно.")

        # Обмен строк
        A[i], A[max_row] = A[max_row], A[i]
        b[i], b[max_row] = b[max_row], b[i]

        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]
            for k in range(i, n):
                A[j][k] -= factor * A[i][k]
            b[j] -= factor * b[i]  # Правильное обновление b

    # Вычисление определителя
    determinant = 1
    for i in range(n):
        determinant *= A[i][i]

    # Обратный ход для нахождения X
    x = [0] * n
    for i in range(n - 1, -1, -1):
        sum_ax = sum(A[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (b[i] - sum_ax) / A[i][i]

    # Вычисляем вектор невязки, используя ОРИГИНАЛЬНУЮ матрицу и вектор b
    residuals = [b_orig[i] - sum(A_orig[i][j] * x[j] for j in range(n)) for i in range(n)]

    return determinant, A, x, residuals, b  # Теперь возвращаем изменённое b


def gaussian_elimination_numpy(A, b):
    """Решение СЛАУ методом Гаусса с выбором главного элемента с использованием NumPy"""
    n = len(A)

    A_orig = copy.deepcopy(A)
    b_orig = b.copy()

    A = A.astype(float)
    b = b.astype(float)

    for i in range(n):
        # Поиск главного элемента
        max_row = i + np.argmax(np.abs(A[i:, i]))
        if A[max_row, i] == 0:
            raise ValueError("Матрица вырожденная, решение невозможно.")

        # Обмен строк
        A[[i, max_row]] = A[[max_row, i]]
        b[[i, max_row]] = b[[max_row, i]]

        # Прямой ход
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]  # Корректное обновление b

    # Вычисление определителя
    determinant = np.prod(np.diag(A))

    # Обратный ход
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

    # Вычисление вектора невязок
    residuals = np.dot(A_orig, x) - b_orig  # Используем оригинальные A и b

    return determinant, A, x, residuals, b  # Теперь возвращаем b
