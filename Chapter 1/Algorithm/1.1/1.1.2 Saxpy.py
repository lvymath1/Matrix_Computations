import numpy as np

def saxpy(a, x, y):
    for i in range(len(x)):
        y[i] = y[i] + a * x[i]
    return y

# Test Examples
a = 2.0
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

result = saxpy(a, x, y)
print(f"SAXPY result for is {result}")