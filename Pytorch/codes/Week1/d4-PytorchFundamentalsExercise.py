# Import torch
import torch 

# 2. Create a random tensor with shape (7, 7).
a = torch.rand(7,7)

# 3. Perform a matrix multiplication on the tensor from 2 with another random tensor with shape (1, 7) (hint: you may have to transpose the second tensor). 
# Create another random tensor
b = torch.rand(1,7)

# Perform matrix multiplication 
a_mul = a.mm(b.T)
print(a_mul)
print(a_mul.shape)


# 4. Set the random seed to 0 and do 2 & 3 over again.
# Set manual seed
import random 
torch.manual_seed(0)

# Create two random tensors
a = torch.rand(7,7)
b = torch.rand(1,7)

# Matrix multiply tensors
a_mul = a.mm(b.T)
print(a_mul)
print(a_mul.shape)



#5. Speaking of random seeds, we saw how to set it with torch.manual_seed() but is there a GPU equivalent? (hint: you'll need to look into the documentation for torch.cuda for this one) If there is, set the GPU random seed to 1234.
# Set random seed on the GPU
torch.cuda.manual_seed(1234)        # seeds the current GPU
torch.cuda.manual_seed_all(1234)    # seeds ALL GPUs (for multi-GPU setups)




# 6. Create two random tensors of shape `(2, 3)` and send them both to the GPU (you'll need access to a GPU for this). Set `torch.manual_seed(1234)` when creating the tensors (this doesn't have to be the GPU random seed)

# Set random seed
torch.manual_seed(1234)

# Check for access to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create two random tensors on GPU
a = torch.rand(2,3).to(device)
b = torch.rand(2,3).to(device)
print("Device:",device)
print(a)
print(b)

# 7. Perform a matrix multiplication on the tensors you created in 6 (again, you may have to adjust the shapes of one of the tensors).
# Perform matmul on tensor_A and tensor_B
a_mul = a.mm(b.T)
print(a_mul)
print(a_mul.shape)


#8. Find the maximum and minimum values of the output of 7.
# Find max
max = a_mul.max()
print(max)
# Find min
min = a_mul.min()
print(min)


#9. Find the maximum and minimum index values of the output of 7.
# Find arg max
argmax = a_mul.argmax()
print(argmax)

# Find arg min
argmin = a_mul.argmin()
print(argmin)


# 10. Make a random tensor with shape `(1, 1, 1, 10)` and then create a new tensor with all the `1` dimensions removed to be left with a tensor of shape `(10)`. Set the seed to `7` when you create it and print out the first tensor and it's shape as well as the second tensor and it's shape.
# Set seed
torch.manual_seed(7)

# Create random tensor
a = torch.rand(1,1,1,10)

# Remove single dimensions
final_a = torch.squeeze(a)

# Print out tensors and their shapes
print(a, a.shape)
print(final_a, final_a.shape)

