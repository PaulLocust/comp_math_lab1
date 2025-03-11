import random


def print_matrix(A):
    for row in A:
        print(" ".join(f"{num:10.4f}" for num in row))


def generate_random_matrix(n, min_val=-10, max_val=10):
    """Генерирует случайную матрицу A размерности n x n и вектор b размерности n."""
    A = [[random.randint(min_val, max_val) for _ in range(n)] for _ in range(n)]
    b = [random.randint(min_val, max_val) for _ in range(n)]
    return A, b
