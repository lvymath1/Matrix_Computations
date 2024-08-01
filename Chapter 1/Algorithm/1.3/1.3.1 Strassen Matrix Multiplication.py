import numpy as np


def strass(A, B, n, n_min):
    if n <= n_min:
        return np.dot(A, B)

    m = n // 2

    # Partition matrices into quadrants
    A11, A12, A21, A22 = A[:m, :m], A[:m, m:], A[m:, :m], A[m:, m:]
    B11, B12, B21, B22 = B[:m, :m], B[:m, m:], B[m:, :m], B[m:, m:]

    # Calculate the seven products, p1 to p7
    P1 = strass(A11 + A22, B11 + B22, m, n_min)
    P2 = strass(A21 + A22, B11, m, n_min)
    P3 = strass(A11, B12 - B22, m, n_min)
    P4 = strass(A22, B21 - B11, m, n_min)
    P5 = strass(A11 + A12, B22, m, n_min)
    P6 = strass(A21 - A11, B11 + B12, m, n_min)
    P7 = strass(A12 - A22, B21 + B22, m, n_min)

    # Combine the intermediate products into the final result
    C11 = P1 + P4 - P5 + P7
    C12 = P3 + P5
    C21 = P2 + P4
    C22 = P1 + P3 - P2 + P6

    # Combine the quadrants into a single matrix
    C = np.empty((n, n), dtype=A.dtype)
    C[:m, :m] = C11
    C[:m, m:] = C12
    C[m:, :m] = C21
    C[m:, m:] = C22

    return C
