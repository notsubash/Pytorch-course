import torch

x = torch.randn(3, requires_grad=True)
print(x)

y = x+2
print(y)

z = y*y*2
#z= z.mean()
print(z)

v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float)
z.backward(v) #dz/dx Gradient of z with respect to x 
## If the tensor is a scalar, you don't need to specify the gradient
print(x.grad)

# Preventing Gradients from tracking
# x.requires_grad_(False)
# x.detach()
# with torch.no_grad():
with torch.no_grad():
    y = x + 2
    print(y)

################################################################

weights = torch.ones(4, requires_grad=True)

for epoch in range(4):
    model_output = (weights*3).sum()

    model_output.backward()

    print(weights.grad)

    weights.grad.zero_()

#################################################################
#Optimization

weights = torch.ones(4, requires_grad=True)
optimizer = torch.optim.sgd(weights, lr=0.01)
optimizer.step()
optimizer.zero_grad()



# Whenever we want to calculate gradients, we need to set the requires_grad flag to True
# We backpropagate the gradients by calling the backward() method on the loss tensor
# We then clear the gradients by calling the zero_grad() method on the optimizer

