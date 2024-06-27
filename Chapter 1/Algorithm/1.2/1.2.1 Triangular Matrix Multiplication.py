import numpy as np


def triangular_matrix_multiplication(A, B, C):
    """
    Overwrites C with C + AB for upper triangular matrices A, B, C.
    """
    n = A.shape[0]
    for i in range(n):
        for j in range(i, n):
            for k in range(i, j + 1):
                C[i, j] += A[i, k] * B[k, j]

# Example usage:
n = 3

# Define upper triangular matrices A, B, and C
A = np.array([[1, 2, 3],
              [0, 4, 5],
              [0, 0, 6]], dtype=float)

B = np.array([[7, 8, 9],
              [0, 10, 11],
              [0, 0, 12]], dtype=float)

C = np.array([[1, 1, 1],
              [0, 1, 1],
              [0, 0, 1]], dtype=float)

print("Original C matrix:")
print(C)

triangular_matrix_multiplication(A, B, C)

print("Updated C matrix:")
print(C)
