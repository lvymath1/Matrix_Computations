import numpy as np

# A is a matrix with m rows and r columns, B is a matrix with r rows and n columns
# and the algorithm time complexity is also O(mn)。
def matrix_multiplication(A, B):
    m, r = A.shape
    r, n = B.shape
    C = np.zeros((m, n), dtype=np.int32)  # Initialize the result matrix C

    for i in range(m):
        for j in range(n):
            for k in range(r):
                C[i, j] += A[i, k] * B[k, j]

    return C

# Test Examples
A = np.array([[1, 2],
              [3, 4],
              [5, 6]])
B = np.array([[7, 8, 9],
              [10, 11, 12]])

result = matrix_multiplication(A, B)
print("Matrix multiplication result:")
print(result)
