import numpy as np

# A is a matrix with m rows and n columns, and the algorithm time complexity is also O(mn)。
def row_oriented_gaxpy(A, x, y):
    for j in range(A.shape[1]):  # Iterate over the rows of matrix A
        for i in range(A.shape[0]):  # Iterate over the columns of matrix A
            y[i] += A[i, j] * x[j]
    return y

def row_oriented_gaxpy_vector(A, x, y):
    for j in range(A.shape[1]):  # Iterate over the rows of matrix A
        y += A[:, j] * x[j]
    return y

# Test Examples
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

result = row_oriented_gaxpy(A, x, y)
print(f"Result is {result}")


A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

result = row_oriented_gaxpy_vector(A, x, y)
print(f"Result is {result}")
