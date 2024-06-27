import numpy as np

def vector_matrix_multiplication(A_vec, x, y):
    """
    Performs the operation y = y + A * x
    for a vectorized representation of a matrix A and vectors x, y.

    Parameters:
    A_vec (np.ndarray): Vectorized representation of the matrix A.
    x (np.ndarray): Vector x.
    y (np.ndarray): Vector y (will be updated in place).
    """
    n = len(x)
    l = 0  # Assuming the indexing starts from 0 in Python

    for j in range(1, n + 1):
        for i in range(1, j):
            index = (i - l) * n - (i * (i - 1)) // 2 + (j - 1)
            y[i - 1] += A_vec[index] * x[j - 1]
        for i in range(j, n + 1):
            index = (j - l) * n - (j * (j - 1)) // 2 + (i - 1)
            y[i - 1] += A_vec[index] * x[j - 1]

# Example usage:
n = 3

# Define the vectorized representation of the matrix A
A_vec = np.array([1, 2, 3, 4, 5, 6], dtype=float)

# Define vector x
x = np.array([1, 2, 3], dtype=float)

# Define vector y (initially zero)
y = np.zeros(n, dtype=float)

print("Original y vector:")
print(y)

vector_matrix_multiplication(A_vec, x, y)

print("Updated y vector:")
print(y)
