import torch # type: ignore
#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#print(torch.__version__)


# SCALARS
scalar = torch.tensor(7) 
print("SCALAR DIM:", scalar.ndim) # Vector dimension is 0
print("SCALAR SHAPE:", scalar.shape) # Vector shape is empty tuple
print(scalar) 
# VECTORS 
vector = torch.tensor([7,7])
print("VECTOR DIM:", vector.ndim) # Vector dimension is 1
print("VECTOR SHAPE:", vector.shape) # Vector shape is the number of elements in the vector
print(vector)
# MATRICES
MATRIX = torch.tensor([[7,8],
                       [9,10]])
print("MATRIX DIM:", MATRIX.ndim) # Matrix dimension is 2
print("MATRIX SHAPE:", MATRIX.shape) # Matrix shape is the number of rows and columns in the matrix
print(MATRIX)   
#TENSORS
# Tensor 
TENSOR = torch.tensor([[[1,2,3],
                      [4,5,6,],
                      [7,8,9]]])
TENSOR
print(TENSOR)
print("TENSOR DIM:", TENSOR.ndim) # Tensor dimension is 3 (number of nested lists)
print("TENSOR SHAPE:", TENSOR.shape) # Tensor shape is the number of elements in each dimension (1, 3, 3) -> 1 tensor, 3 rows, 3 columns
#----------------



# Random Tensors , We can use torch.rand() to create random tensors of a specific shape. The values in the tensor will be between 0 and 1.
random_tensor = torch.rand(3,4) # 3 rows, 4 columns
print(random_tensor) # prints a random tensor of shape (3,4) with values between 0 and 1 

# Create a random tensor with similar shape to an image tensor 
random_image_size_tensor = torch.rand(size=(224,224,3)) # height, width, color channels (RGB)
print(random_image_size_tensor.shape) # prints the shape of the random image size tensor (224, 224, 3)
print(random_image_size_tensor.ndim) # prints the dimensionality of the random image size tensor

#Zeros and Ones , We can use torch.zeros() and torch.ones() to create tensors filled with zeros and ones, respectively.
zeros = torch.zeros(size=(3,4)) # creates a tensor of shape (3,4) filled with zeros
print(zeros) # prints the tensor of zeros
ones = torch.ones(size=(3,4)) # creates a tensor of shape (3,4) filled with ones
print(ones) # prints the tensor of ones 



# Range of tensors , We have to use torch.arange() to create a tensor with a range of values.

zero_to_ten = torch.arange(0,10) # creates a tensor with values from 0 to 9
print(zero_to_ten) # prints the tensor with values from 0 to 9

# Tensor datatypes , We can specify the datatype of a tensor using the dtype argument when creating a tensor.
# Default datatype for tensors is float32
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None, # What datatype is the tensor ( e.g. float32 or float 16)
                               device=None, # What device is your tensor on 
                               requires_grad=False) # if True, operations perfromed on the tensor are recorded 

print(float_32_tensor.shape, float_32_tensor.dtype, float_32_tensor.device) # prints the shape, datatype, and device of the float_32_tensor

