import numpy as np

# Example matrix
matrix = np.array([[1, 2],
                   [3, 4]])

# Calculate the inverse of the matrix
inverse_matrix = np.linalg.inv(matrix)

# Print the result
print("Original matrix:")
print(matrix)
print("Inverse matrix:")
print(inverse_matrix)
