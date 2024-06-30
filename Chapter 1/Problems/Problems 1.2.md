Answer

### P 1.2.1

需要一个额外的矩阵来存储中间结果，然后再将其复制回矩阵，当为普通矩阵时空间复杂度 $n^2$，当为上三角矩阵时空间复杂度 $\frac{n(n+1)}{2}$.

### P 1.2.2

一个Hessenberg有 $\frac{n^2}{2} + \frac{3n}{2} + 1$ 个非零项，算第一列只需要从后向前乘，那么前 $r - 1$ 个矩阵有 $(r - 1) \cdot (\frac{n^2}{2} + \frac{3n}{2} + 1)$ 个乘法。
每个矩阵还要先算一下减法，那么一共 $r \cdot n$ 个减法。乘法和减法加起来有 $(r - 1) \cdot (\frac{n^2}{2} + \frac{3n}{2} + 1) + r \cdot n$ 个flop。


### P 1.2.3

用矩阵ijk矩阵乘法进行计算，因为A的第i行$[i, n-1]$和B的第j列$[0,j]$是非零的，那么仅需要计算$[i,j]$之间上的数即可。
```python
def matrix_multiplication(A, B, C):
    n, _ = A.shape
    for i in range(n):
        for j in range(n):
            for r in range(i, j + 1):
                C[i][j] += A[i][r] * B[r][j]
```

### P 1.2.4

跟算法1.2.2几乎一样，只不过需要更改一些范围。

```python
def band_matrix_vector_multiplication(A_band, x, y, p, q):
    """
    Performs the operation y = y + Ax
    for banded matrix A_band and vectors x, y.
    """
    m = len(y)
    n = len(x)

    for j in range(n):
        a1 = max(0, j - q)
        a2 = min(m, j + p + 1)
        beta1 = max(0, q - j)
        beta2 = beta1 + (a2 - a1)

        y[a1:a2] += A_band[beta1:beta2, j] * x[j]
```

### P 1.2.5

计算 $z$ 的实部与虚部公式为

$Re(z) = Re(A)Re(z) - Im(A)Im(x)$

$Im(z) = Re(A)Im(z) + Im(A)Re(x)$

因为根据Hermitian矩阵做了特定数据结构，所以代码中需要认真讨论$Re(A)$ 和 $Im(A)$, 在 $A.herm$ 中所对应的位置。

```python
import numpy as np

def matrix_vector_multiplication(A_herm, re_x, im_x):
    n = A_herm.shape[0]
    re_z = np.zeros(n, dtype=float)
    im_z = np.zeros(n, dtype=float)

    for i in range(n):
        for j in range(n):
            if j < i:
                re_z[i] += A_herm[j][i] * re_x[j] - A_herm[i][j] * im_x[j]
                im_z[i] += A_herm[j][i] * im_x[j] + A_herm[i][j] * re_x[j]
            elif j == i:
                re_z[i] += A_herm[j][i] * re_x[j]
                im_z[i] += A_herm[j][i] * im_x[j]
            else:
                re_z[i] += A_herm[i][j] * re_x[j] + A_herm[j][i] * im_x[j]
                im_z[i] += A_herm[i][j] * im_x[j] - A_herm[j][i] * re_x[j]
    return re_z, im_z


A_herm = np.array([[1, 2, 3],
                   [1, 3, 4],
                   [2, 3, 5]])

re_x = np.array([1, 2, 3])
im_x = np.array([4, 5, 6])

re_z, im_z = matrix_vector_multiplication(A_herm, re_x, im_x)
print("Real part of result:", re_z)
print("Imaginary part of result:", im_z)
```

### P 1.2.7

把算法 1.1.3 的gaxpy中的 $A[i][j]$ 改成 $a[abs(i-j)]$ 即可。

```python
import numpy as np


def matrix_vector_gaxpy(A_vec, x, y):
    """
    Performs the operation y = y + A * x
    for a vectorized representation of a matrix A and vectors x and y.
    """
    n = A_vec.shape[0]
    for i in range(n):
        for j in range(n):
            y[i] += A_vec[abs(j - i)] * x[j]


# Example usage:
n = 3

# Define the vectorized representation of the matrix A
A_vec = np.array([1, 2, 3, 4], dtype=float)

# Define vectors x and y
x = np.array([1, 2, 3, 4], dtype=float)
y = np.array([4, 5, 6, 7], dtype=float)

# Perform the operation
matrix_vector_gaxpy(A_vec, x, y)

# Output the result
print(f"Updated y vector: {y}")

```
