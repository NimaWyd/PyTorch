import torch # type: ignore
import numpy as np 


#creating tensor 
x = torch.empty(2,3) # 2x3 Matrix 
print(x)

# creating tensor with random values 
x = torch.rand(2,2)
print(x)

# zero tensors
x = torch.zeros(2,2)
print(x)
print(x.dtype) # to get the type of the values

# changing types
x = torch.rand(2,2,dtype = torch.float16)
print(x.dtype)


x = torch.tensor([2.5,0.1])
print(x)


# Basic Operations 
x = torch.rand(2,2)
y = torch.rand(2,2)
print(x)
print(y)
z = x + y  # z = torch.add(x,y)    -     y.add_(x) - they all do the same thing
print(z)  
z = x - y  # z = torch.sub(x,y)         
print(z)
z = x * y  # z = torch.mul(x,y)
print(z)
z = x / y  # z = torch.div(x,y)
print(z)



x = torch.rand(3,5)
print(x)
print(x[:,0])  # print all rows of first column
print(x[1,:])  # print all the elements in second row
# we can use x[1,1].item if we are pointing to a single element

# Reshaping
x = torch.rand(4,4)
print(x)
y = x.view(16)   # we can use -1 as an input so torch itself will determine the size 
print(y)


# convert from np to torch and vice versa 

# Tensor to Numpy
a = torch.ones(5)
print(a)
b = a.numpy()  # numpy can only handle cpu tensors and it gives and error for gpu tensors(cuda)
print(type(b))
print(b)
a.add_(1)
print(a)
print(b) # change in a will make a change on b  because they are both pointing into the same memory location , it only happens if the tensor is on CPU 

# Numpy to Tensor
import numpy as np
a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)
np.add(a, 1, out=a)
print(a)
print(b)


x = torch.ones(5, requires_grad=True) # by default requires_grad is False, track all operations on this tensor so I can compute gradients with respect to it later 
print(x)

