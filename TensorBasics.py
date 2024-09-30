import torch
import numpy as np

#empty tensor
x = torch.empty(2, 2,2, 3)
print(x)

#random tensor
x = torch.rand(2, 2,2, 3)
print(x)

#zeros
x = torch.zeros(2,2)
print(x)

#ones
x = torch.ones(2,2)
print(x)

#dtype
print(x.dtype)

#setting dtype
x = torch.rand(2,2, dtype=torch.float16)
print(x)

#size of tensor
print(x.size())

#torch from python list
x = torch.tensor([2.5, 0.1])
print(x)

x = torch.rand(2,2)
y = torch.rand(2,2)
print(x)
print(y)

#Addition of tensors
z = x + y 
print(z)

z = torch.add(x, y)
print(z)

#In place addition
y.add_(x)
print(y)

#Subtraction
z = torch.sub(x, y) 
print(z)

#Multiplication
z = x * y
z = torch.mul(x, y)
print(z)

#In place multiplication
x.mul_(y)
print(x)

#Division
z = torch.div(x, y)
z = x/y
print(z)

############################################

#Slicing
x = torch.rand(5, 3)
print(x)
print(x[:, 0]) #All rows, only first column 0
print(x[1, :]) #Only row 1, all columns

#Single item from the tensor
print(x[1, 1].item())

#Reshaping
x = torch.rand(4, 4)
print(x)
y = x.view(16)
print(y.size())
print(y)

y = x.view(-1, 8)
print(y)
print(y.size())


#Converting to numpy
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)
print(type(b))

#Tensors and Numpy both share the same memory location, so if you change one, the other will change
#Be careful while performing any inplace operations in them or making changes to any of them
a.add_(1)
print(a)
print(b)

#Converting numpy to tensor
a = np.ones(5)
print(a)

b = torch.from_numpy(a) #Default dtype is float64
print(b)
print(b.dtype)

a +=1
print(a)
print(b)

#Working on the GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device)
    y = torch.ones(5)
    y = y.to(device)
    z = x + y
    z = z.to("cpu")
    print(z)

#Requires Gradient
x = torch.ones(5, requires_grad=True)
print(x)