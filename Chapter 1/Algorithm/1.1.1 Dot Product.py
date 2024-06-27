import numpy as np

def dot_product(x, y):
    c = 0
    for i in range(len(x)):  # len(x) 或 len(y) 都是 n
        c += x[i] * y[i]
    return c

# Test Examples
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

result = dot_product(x, y)
print(f"Dot product of {x} and {y} is {result}")