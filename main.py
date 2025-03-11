import numpy as np
from gauss import gaussian_elimination_pure, gaussian_elimination_numpy
from io_handler import read_matrix, write_results
from utils import print_matrix, generate_random_matrix
import copy  # Для глубокого копирования


def main():
    print("Выберите источник данных:")
    print("1 - Ввести с клавиатуры")
    print("2 - Загрузить из файла")
    print("3 - Сгенерировать случайную матрицу")

    choice = input("Введите номер (1/2/3): ")

    if choice == "1":
        A, b = read_matrix(from_file=False)
    elif choice == "2":
        filename = input("Введите имя файла: ")
        A, b = read_matrix(from_file=True, filename=filename)
    elif choice == "3":
        while True:
            n = int(input("Введите размерность матрицы (n <= 20): "))
            if 1 <= n <= 20:
                break
            print("Ошибка: размерность должна быть от 1 до 20.")
        A, b = generate_random_matrix(n)
    else:
        print("Ошибка: неверный ввод.")
        return

    # Создаём копии исходных данных
    A_orig = copy.deepcopy(A)
    b_orig = b.copy()

    print("\nМатрица A:")
    print_matrix(A)
    print("\nb:", b)

    # Выбираем метод решения
    print("\nВыберите метод решения:")
    print("1 - Метод Гаусса (чистый Python)")
    print("2 - Метод Гаусса (NumPy)")

    method_choice = input("Введите номер (1/2): ")

    if method_choice == "1":
        det, U, x, residuals, b_transformed = gaussian_elimination_pure(A, b)
    elif method_choice == "2":
        det, U, x, residuals, b_transformed = gaussian_elimination_numpy(np.array(A), np.array(b))
    else:
        print("Ошибка: неверный ввод.")
        return

    print("\nОпределитель матрицы:", det)
    print("\nТреугольная матрица (после приведения):")
    print_matrix(np.column_stack((U, b_transformed)))
    print("\nВектор решений:")
    print(x)
    print("\nВектор невязок:")
    print(residuals)

    # Передаём изменённый b_transformed в write_results
    write_results(A_orig, b_orig, det, U, x, residuals, b_transformed)


if __name__ == "__main__":
    main()
