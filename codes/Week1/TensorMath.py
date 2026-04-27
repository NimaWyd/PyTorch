import torch
import numpy as np


#defining a tensor
device = "cuda" if torch.cuda.is_available() else "cpu" # in this way we can use cuda if we have gpu
my_tensor = torch.tensor([[1,2,3],
                          [4,5,6]],
                         dtype= torch.float32,
                         device= device, 
                         requires_grad=True) #optional
print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)

#common initialization methods 
x = torch.empty(size=(3, 3))
print(x)
x = torch.zeros(3,3)
print(x)
x = torch.rand(3,3)
print(x)
x = torch.ones(3,3)
print(x)
x = torch.eye(5,5) # I / Eye Matrix
print(x)


# Range in torch
x = torch.arange(start=0, end=5, step=1)
print(x)
x = torch.linspace(start=0.1, end=1, steps=10) #initialize 10 values between 0.1 and 1
print(x)
x = torch.empty(size=(1,5)).normal_(mean=0, std=1) #initialize a tensor with normal distribution
print(x)
x = torch.diag(torch.ones(3)) # works same way as I/Eye func
print(x)


# how to convert tensors to other types 

tensor = torch.arange(4)
print(tensor)
print(tensor.bool()) # create a False/True tensor , 0 is False and the rest are True
print(tensor.short()) #turns numbers into int16
print(tensor.long()) # turns numbers into int64
print(tensor.half()) # turns numbers into float16
print(tensor.float()) # turns numbers into float32
print(tensor.double()) # turns numbers into float64



# how to convert numpy array to tensor and vice-versa
np_array = np.zeros((5,5))
tensor = torch.from_numpy(np_array)
np_array_back = tensor.numpy()


from numpy.matrixlib.defmatrix import matrix
# MATH OP
x = torch.tensor([1,2,3])
y = torch.tensor([9,8,7])

# Addition
z1 = torch.empty(3) # 1st way
torch.add(x,y,out=z1)
print(z1)
z2 = torch.add(x,y) # 2nd way
print(z2)
z = x + y # 3rd way (preferred way)
print(z)

# Subtraction
z = x - y
print(z)

# Division
z = torch.true_divide(x,y) # element wise division , y can be integer as well
print(z)

# Inplace operations
t = torch.zeros(3)
t.add_(x)  # when we have _ after our operation it means our operation will happen in place and it doesn't create a copy and its good for complexity
t += x # another way to do the inplace operation t = t + x 

# Exponentiation
z = x.pow(2)
print(z)
z = x ** 2 # preferred way

# Simple comparison 
z = x > 0 
print(z)
z = x < 0
print(z)

# Matrix Multiplication
x1 = torch.rand((2,5))
x2 = torch.rand((5,3))
x3 = torch.mm(x1,x2) # 2nd way  2x3 Matrix
x3 = x1.mm(x2) # 1st way
print(x3)

# Matrix exponentiation
matrix_exp = torch.rand(5,5)
print(matrix_exp.matrix_power(3))

# Element wise mult
z = x * y
print(z)

# Dot product
z = torch.dot(x,y)
print(z)