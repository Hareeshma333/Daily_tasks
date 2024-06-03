import numpy as np

# Create a NumPy array
arr = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])

# Get unique values and their counts
unique_values, counts = np.unique(arr, return_counts=True)

# Zip the unique values and their counts together
freq_dict = dict(zip(unique_values, counts))

print("Frequency of unique values:")
for value, count in freq_dict.items():
    print(f"{value}: {count}")
