array = [1, 0, 2, 3, 0, 4, 0]
count = sum(1 for x in array if x != 0)
print(count)  # Output: 4
