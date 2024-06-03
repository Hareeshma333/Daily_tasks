import numpy as np

# Example matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Using dot() function
result = np.dot(A, B)
print(result)

# Using @ operator
result = A @ B
print(result)

# Using matmul() function
result = np.matmul(A, B)
print(result)