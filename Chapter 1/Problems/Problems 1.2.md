# Answer

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

### P 1.2.6

对称矩阵 $A[i][j]$ 当 $j < i$ 时对应 $A_vec[j * n - (j * (j + 1)) // 2 + i]$, 把这一关系带入到 $B = X^TAX$ 中, 
可以发现 $b_{ij} = \sum_{k = 1}^{n}\sum_{l=1}^{n} x_{il}a_{im}x_{mj}$.

```python
import numpy as np


def vector_matrix_multiplication(A_vec, X):
    """
    Performs the operation y = y + A * x
    for a vectorized representation of a matrix A and vectors x, y.
    """
    n, p = X.shape
    B_vec = np.zeros(p * (p + 1) // 2)

    for i in range(p):
        for k in range(n):
            for l in range(n):
                for j in range(i, p):
                    B_vec[j * n - (j * (j + 1)) // 2 + i] += X[i][l] * A_vec[k * n - (k * (k + 1)) // 2 + i] * X[k][j]
    return B_vec
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

### P 1.2.8

把上一问的gaxpy中的 $abs(j - i)$ 改成 $(i + j) % n$ 即可，几乎同理。

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
            y[i] += A_vec[(i + j) % n] * x[j]


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

### P 1.2.9

对称矩阵那么仅需要保留对角线之上的矩阵就可以。

```python
import numpy as np


def symmetric_band_matrix_gaxpy(A_band, x, y):
    """
    Performs the operation y = y + Ax
    for banded matrix A_band and vectors x, y.
    """
    p, n = A_band.shape
    band_width = p - 1
    for i in range(n):
        left = max(0, i - band_width)
        right = min(n - 1, i + band_width)
        for j in range(left, i+1):
            y[i] += A_band[band_width - (i - j)][i] * x[j]
        for j in range(i+1, right+1):
            y[i] += A_band[band_width - (j - i)][j] * x[j]


# Example usage:
n = 4

# Define a banded matrix A_band
# A_band is a 2D array where rows represent the diagonals of the banded matrix
A_band = np.array([[0, 1, 3, 5],
                   [1, 2, 4, 6]])

# Define vector x
x = np.array([1, 2, 3, 4], dtype=float)

# Define vector y (initially zero)
y = np.zeros(n, dtype=float)

print("Original y vector:", y)

symmetric_band_matrix_gaxpy(A_band, x, y)

print("Updated y vector:", y)
```

### P 1.2.10

$$(A + uv^T)^k = [A^{k-1}u, ..., Au, u] 
\begin{bmatrix} 
v^T \\ 
... \\ 
v^T(A + uv^T)^{k-2} \\ 
v^T(A + uv^T)^{k-1} 
\end{bmatrix}$$

所以这里 \(X = [A^{k-1}u, ..., Au, u]\)，\(Y = [v^T, ..., v^T(A + uv^T)^{k-2}, v^T(A + uv^T)^{k-1}]\)。

计算 \(u, Au, ...., A^{k-1}u\) 一共需要 \(r \cdot n^2\) 次浮点运算（flops）。

计算 \(v^T, ..., v^T(A + uv^T)^{k-2}, v^T(A + uv^T)^{k-1}\) 一共需要 \(2 n^2\) 次浮点运算计算 \((A + uv^T)\)，又需要 \(r \cdot n^2\) 次浮点运算。

所以总共有 \((2r + 2) \cdot n^2\) 次浮点运算。

### P 1.2.10

$D_n$ 左乘的含义是把向量往下推一次, 那么 $D_n^k$ 就是把向量往下推k个. 那么推完之后 $n-k, n-1$ 位置在前面, $0, n - k - 1$ 位置在之后

```python
import numpy as np


def upshif_k_times(x, k):
    """
    Performs the operation y = y + Ax
    for banded matrix A_band and vectors x, y.
    """
    n = x.shape[0]
    y = np.zeros([n, 1])
    for i in range(n):
        y[(i + k) % n] = x[i]
    return y

```

### P 1.2.12

显然，但是证明比较啰嗦，读者可以自行证明。


### P 1.2.13

有趣的排列组合题目, 要求的是1到n的全排列，排列要求要不然第i位就是i，如果i位是j的话，那第j位必须是i，问一共有多少种情况。

这里分类讨论，如果第n位为n，那么问题转化为n-1的情况。如果第n位为j，那么j位置必须为n，问题转化为n-2中情况，j有n-1种选法。

假设1到n满足结果的有 $A(n)$ 种, 那么 $A(n) = A(n-1) + (n-1)\cdot A(n-2)$, 其中 $A(1) = 1, A(2) = 2$.