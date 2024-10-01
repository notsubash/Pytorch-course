### Back Propagation implementation from scratch

# Example variables:
# x = 1
# y = 2
# w = 1

import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True) #Cause we are interested in the gradient

#forward pass and compute the loss
y_hat = w * x
loss = (y_hat - y)**2

print(loss)

#Backward pass
#Pytroch computes local gradients and backward pass automatically for us
loss.backward()
print(w.grad)

#### Next steps:
#### update weights
### next forward and backward pass


