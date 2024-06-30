import numpy as np

def vector_matrix_multiplication(A_vec, x, y):
    """
    Performs the operation y = y + A * x
    for a vectorized representation of a matrix A and vectors x, y.
    """
    n = len(x)

    for j in range(n):
        for i in range(j):
            index = i * n - (i * (i + 1)) // 2 + j
            y[i] += A_vec[index] * x[j]
        for i in range(j, n):
            index = j * n - (j * (j + 1)) // 2 + i
            y[i] += A_vec[index] * x[j]

# Example usage:
n = 3

# Define the vectorized representation of the matrix A
A_vec = np.array([1, 2, 3, 4, 5, 6], dtype=float)

# Define vector x
x = np.array([1, 2, 3], dtype=float)

# Define vector y (initially zero)
y = np.zeros(n, dtype=float)

print("Original y vector:", y)

vector_matrix_multiplication(A_vec, x, y)

print("Updated y vector:", y)
