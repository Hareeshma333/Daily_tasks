import numpy as np
# create a one dimentional array
array_1d = np.array([1,2,3,4,5,6])
print("array1d:",array_1d)
print("shape of array1d:",array_1d.shape)
#reshape the 1d array into 2d array
array_2d = array_1d.reshape((2,3))
print("2d array :",array_2d)
print("shape of array_2d:",array_2d.shape)
#reshape the 1d array into 3d array
array_3d = array_1d.reshape((3,2))
print("3d array :",array_3d)
print("shape of array_3d:",array_3d.shape)

#reshape back a 2D array to 1D
array_1d_back=array_2d.reshape(6)
print(array_1d_back)