# Answer

(基本都是等式证明)

### P 1.3.1
假设第 i 行高度为 $m_i$, 第 j 列宽度为 $n_i$
发现 $A_{ij}^{xy}$ 的位置在 $\sum_{s=1}^{i-1} m_s + x$, 在 $\sum_{s=1}^{j-1} n_s + y$,
转置之后的 $A_{ij}^{xy}$ 的位置就在 $\sum_{s=1}^{j-1} n_s + y$ 行, $\sum_{s=1}^{i-1} m_s + x$ 列。
发现两者正好呈现转置的关系.

### P 1.3.2
根据哈密顿矩阵，先算一下 $M^2$

$$
\begin{bmatrix} 
A^2 + GF & \quad & AG-GA^T\\
FA-A^TF  & \quad & FG+A^TA^T\\
\end{bmatrix}
$$

这里我们发现左上和右下的矩阵呈现转置的关系。右上和左下都是对角矩阵。
所以左上的矩阵仅需要 $2\cdot\frac{n}{2}^3$, 左上和右下分别需要 $\frac{n^2(n+2)}{16}$, 最终有 $\frac{n^3}{2}$ 个flop.

### P 1.3.3
$A$ 是反对角的对称的, 可以计算下标证明。

### P 1.3.4


### P 1.3.5
可以证明矩阵每一行相加都是1，同理矩阵每一列相加也是1. 矩阵是一个01矩阵。

### P 1.3.6
讨论清楚即可。

### P 1.3.7
$xy^T$ 在第i行第j列为 $x_i \cdot y_j$, $vec(xy^T)$ 第 $i * m + j$ 为 $x_i \cdot y_j$.
这个结果和 $y \otimes x$ 相等。

### P 1.3.8
