import torch
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
