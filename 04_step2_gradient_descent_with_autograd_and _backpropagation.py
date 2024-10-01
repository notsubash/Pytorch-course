#### Step 2
# Prediction: Manually
# Gradients Computation: Autograd
# Loss Computation: Manually
# Parameter updates: Manually

import numpy as np
import torch
# f = w * x # Not caring about bias here 

# f = 2 * x

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True) #Initializing weights

# Model Prediction
def forward(x):
    return w * x

# Loss = MSE
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()


print(f'Prediction before training: f(5) = {forward(5):.3f}')

# Training
learning_rate = 0.01
n_iters = 100

#Training loop
for epoch in range(n_iters):
    #prediction = foward pass
    y_pred = forward(X)

    #loss
    l = loss(Y, y_pred)

    #gradients = backward pass
    l.backward() #dl/dw

    #update weights
    with torch.no_grad():
        w -= learning_rate * w.grad

    #Zero the gradients
    w.grad.zero_()

    if epoch % 10 ==0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')