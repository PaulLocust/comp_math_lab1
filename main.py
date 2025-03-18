import numpy as np
from gauss import gaussian_elimination_pure, gaussian_elimination_numpy
from io_handler import read_matrix, write_results
from utils import print_matrix, generate_random_matrix
import copy  # Для глубокого копирования


def main():
    while True:  # Основной цикл программы
        try:
            print("\nВыберите источник данных:")
            print("1 - Ввести с клавиатуры")
            print("2 - Загрузить из файла")
            print("3 - Сгенерировать случайную матрицу")
            print("4 - Выйти из программы")

            choice = input("Введите номер (1/2/3/4): ")

            if choice == "4":
                print("Выход из программы.")
                break  # Выход из цикла и завершение программы

            if choice == "1":
                while True:  # Новый цикл для ввода размерности и матрицы
                    try:
                        n = int(input("Введите размерность матрицы (n <= 20): "))
                        if n > 20:
                            raise ValueError("Размерность матрицы должна быть не более 20!")

                        A = np.zeros((n, n))
                        b = np.zeros(n)

                        print(
                            "Введите коэффициенты матрицы A построчно (разделяя числа пробелами, можно использовать запятую или точку для дробных чисел):")
                        for i in range(n):
                            A[i] = list(map(lambda x: float(x.replace(",", ".")), input().split()))

                        print("Введите вектор b (можно использовать запятую или точку для дробных чисел):")
                        b[:] = list(map(lambda x: float(x.replace(",", ".")), input().split()))

                        break  # Выход из цикла, если всё прошло успешно
                    except Exception as e:
                        print(f"Произошла ошибка: {e}")
                        print("Пожалуйста, попробуйте снова.")

            elif choice == "2":
                while True:  # Новый цикл для ввода имени файла
                    filename = input("Введите имя файла: ")
                    A, b = read_matrix(from_file=True, filename=filename)
                    if A is None or b is None:  # Проверяем, произошла ли ошибка
                        print("Попробуйте снова ввести имя файла.")
                    else:
                        break  # Если файл прочитан успешно, выходим из цикла

            elif choice == "3":
                while True:
                    try:
                        n = int(input("Введите размерность матрицы (n <= 20): "))
                        if 1 <= n <= 20:
                            break
                        print("Ошибка: размерность должна быть от 1 до 20.")
                    except ValueError:
                        print("Ошибка: введите целое число.")
                A, b = generate_random_matrix(n)

            else:
                print("Ошибка: неверный ввод. Пожалуйста, выберите 1, 2, 3 или 4.")
                continue  # Начинаем цикл заново

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
                print("Ошибка: неверный ввод. Пожалуйста, выберите 1 или 2.")
                continue  # Начинаем цикл заново

            print("\nОпределитель матрицы:", det)
            print("\nТреугольная матрица (после приведения):")
            print_matrix(np.column_stack((U, b_transformed)))
            print("\nВектор решений:")
            print([round(float(num), 4) for num in x])
            print("\nВектор невязок:")
            print(residuals)

            # Передаём изменённый b_transformed в write_results
            write_results(A_orig, b_orig, det, U, x, residuals, b_transformed)

        except Exception as e:  # Обработка всех исключений
            print(f"Произошла ошибка: {e}")
            print("Пожалуйста, попробуйте снова.\n")


if __name__ == "__main__":
    main()
