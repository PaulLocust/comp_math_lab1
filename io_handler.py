import os
import numpy as np

DATA_DIR = "data"
RESULTS_DIR = "results"


def read_matrix(from_file=True, filename=""):
    """Читает матрицу из файла или с клавиатуры, поддерживая числа с запятой и точкой"""
    if from_file:
        filepath = os.path.join(DATA_DIR, filename)
        try:
            with open(filepath, "r") as file:
                n = int(file.readline().strip())
                if n > 20:
                    raise ValueError("Размерность матрицы должна быть не более 20!")

                # Читаем содержимое файла, заменяя запятые на точки
                lines = [line.replace(",", ".") for line in file]
                data = np.loadtxt(lines)

                A, b = data[:, :-1], data[:, -1]
            return A, b
        except FileNotFoundError:
            print(f"Ошибка: файл '{filename}' не найден. Попробуйте снова.")
            return None, None  # Возвращаем None, чтобы обработать ошибку в главной программе
        except Exception as e:
            print("Ошибка при чтении файла:", e)
            return None, None  # Возвращаем None, чтобы обработать ошибку в главной программе
    else:
        while True:
            try:
                n = int(input("Введите размерность матрицы (n <= 20): "))
                if n > 20:
                    raise ValueError("Размерность матрицы должна быть не более 20!")
                break
            except ValueError as e:
                print("Ошибка:", e)

        A = np.zeros((n, n))
        b = np.zeros(n)

        print(
            "Введите коэффициенты матрицы A построчно (разделяя числа пробелами, можно использовать запятую или точку для дробных чисел):")
        for i in range(n):
            A[i] = list(map(lambda x: float(x.replace(",", ".")), input().split()))

        print("Введите вектор b (можно использовать запятую или точку для дробных чисел):")
        b[:] = list(map(lambda x: float(x.replace(",", ".")), input().split()))

        return A, b

def write_results(A_orig, b_orig, determinant, U, x, residuals, b_transformed, filename="results.txt"):
    """Сохраняет результаты в файл в папке results/"""
    os.makedirs(RESULTS_DIR, exist_ok=True)  # Создание папки, если её нет
    filepath = os.path.join(RESULTS_DIR, filename)

    with open(filepath, "w") as file:
        # Сохраняем исходную матрицу и вектор b
        file.write("Исходная матрица:\n")
        np.savetxt(file, np.column_stack((A_orig, b_orig)), fmt="%.6f")

        file.write("\nОпределитель: {:.6f}\n".format(determinant))

        file.write("\nТреугольная матрица:\n")
        np.savetxt(file, np.column_stack((U, b_transformed)), fmt="%.6f")  # Теперь тут b_transformed

        file.write("\nВектор решений:\n")
        np.savetxt(file, x, fmt="%.6f")

        file.write("\nВектор невязок:\n")
        np.savetxt(file, residuals, fmt="%.6f")

        print(f"\nРезультаты сохранены в {filepath}")
