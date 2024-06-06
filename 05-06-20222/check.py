import numpy as np

array=np.array([10,20,12,15,14,17])
#np.where(array ==20)
#where is the method which check the particular condition for filter and condition

#element greater that 15 for above array

elements =np.where(array>15, 0, array)
print(array)
# print(array[elements])