### Linear Regression from scratch

#### Step 1
# Prediction: Manually
# Gradients Computation: Manually
# Loss Computation: Manually
# Parameter updates: Manually

import numpy as np

# f = w * x # Not caring about bias here 

# f = 2 * x

X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)

w = 0.0 #Initializing weights

# Model Prediction
def forward(x):
    return w * x

# Loss = MSE
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()

# Gradient
# MSE = 1/N * (w*x - y)**2
# Derivative of J with respect to w
# dJ/dW =  1/N 2x (w*x -y)
def gradient(x, y, y_predicted):
    return np.dot(2*x, y_predicted - y).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')

# Training
learning_rate = 0.01
n_iters = 20

#Training loop
for epoch in range(n_iters):
    #prediction = foward pass
    y_pred = forward(X)

    #loss
    l = loss(Y, y_pred)

    #gradients
    dw = gradient(X, Y, y_pred)

    #update weights
    w -= learning_rate * dw

    if epoch % 2 ==0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')