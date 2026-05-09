import torch # type: ignore
import numpy as np # type: ignore


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


from numpy.matrixlib.defmatrix import matrix  # type: ignore
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

# Batch Matrix Multi
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2) # (batch, n, p)
print(out_bmm)

# Example of boradcasting 
x1 = torch.rand((5,5))
x2 = torch.rand((1,5))
print(x1)
print(x2)
z = x1 - x2
print(z)
z = x1 ** x2
print(z)


# Other useful tensor Opes
sum_x = torch.sum(x, dim=0)
print(sum_x)
values, indices = torch.max(x, dim=0)
print(values, indices)
values, indices = torch.min(x, dim=0)
print(values, indices)
abs_x = torch.abs(x) # absolute value of x
print(abs_x)
z = torch.argmax(x, dim=0)   # same as torch.max but this one only returns the index of maximum
print(z)
z = torch.argmin(x, dim=0)
print(z)

mean_x = torch.mean(x.float(), dim=0)
print(mean_x)
z = torch.eq(x,y) # element wise comparison
print(z)


sorted_y,indices = torch.sort(y,dim = 0 , descending=False)
print(sorted_y)

z = torch.clamp(x, min = 0)
print(z)


z = torch.any() # any value = 1 make our expression true 
z = torch.all() # we neee all valuees to = 1 to make our expression true


#Tensor indexing
batch_size = 10
features = 25
x = torch.rand(batch_size, features)
print(x[0].shape) # x[0,:]
print(x[:,0].shape)
print(x[2,0:10]) # to get the third example in batch and get first 10 features
x[0,0] = 100     #assigning value
x = torch.arange(10)
indices = [2,5,8]
print(x[indices])


x = torch.rand(3, 5)
rows = torch.tensor ([1,0])
cols = torch.tensor ([4,0])
print(x[rows,cols])

# More advanced indexing
x = torch.arange(10)
print(x[(x < 2) | (x > 8)])
print(x[x.remainder(2) == 0])

# Useful operations
print(torch.where(x > 5, x, x*2)) # if x is less than 5 it will change it into x*2 and if it is >5 it will be x 
print(torch.tensor([0,0,1,2,2,3,4]).unique())  # we will get the unique values from the tensor
print(x.ndimension())  # its going to check how many dimension of x do we have, if we had 5x5x5 tensor we would have 3 as our output
print(x.numel()) # number of elements

# Tensor reshaping
x = torch.arange(9)
print(x)
x_3x3 = x.view(3,3)
x_3x3 = x.reshape(3,3) # they are similar ,reshape is the safebet 
print(x_3x3)

x1 = torch.rand((2,5))
x2 = torch.rand((2,5))
print(torch.cat((x1,x2),dim = 0).shape)
print(torch.cat((x1,x2),dim = 1).shape)

z = x1.view(-1)
print(z.shape)

batch = 64 
x = torch.rand((batch,2,5))
z = x.view(batch,-1) 
print(z.shape)

z = x.permute(0,2,1)
print(z.shape)

x = torch.arange(10) #[10]
print(x.unsqueeze(0).shape)
print(x.unsqueeze(1).shape)
print(x.unsqueeze(0).unsqueeze(1).shape)

