import numpy as np

def band_matrix_vector_multiplication(A_band, x, y, p, q):
    """
    Performs the operation y = y + Ax
    for banded matrix A_band and vectors x, y.
    """
    n = len(x)

    for j in range(n):
        a1 = max(0, j - q)
        a2 = min(n, j + p + 1)
        beta1 = max(0, q - j)
        beta2 = beta1 + (a2 - a1)

        y[a1:a2] += A_band[beta1:beta2, j] * x[j]

# Example usage:
n = 5
p = 1
q = 1

# Define a banded matrix A_band
# A_band is a 2D array where rows represent the diagonals of the banded matrix
A_band = np.array([[0, 1, 2, 3, 4],  # Lower diagonal (q)
                   [1, 2, 3, 4, 5],  # Main diagonal
                   [1, 2, 3, 4, 0]])  # Upper diagonal (p)

# Define vector x
x = np.array([1, 2, 3, 4, 5], dtype=float)

# Define vector y (initially zero)
y = np.zeros(n, dtype=float)

print("Original y vector:", y)

band_matrix_vector_multiplication(A_band, x, y, p, q)

print("Updated y vector:", y)

y = np.zeros(n, dtype=float)
A = np.array([[1, 1, 0, 0, 0],
              [1, 2, 2, 0, 0],
              [0, 2, 3, 3, 0],
              [0, 0, 3, 4, 4],
              [0, 0, 0, 4, 5]])
print("Updated y vector:", y + np.dot(A, x))