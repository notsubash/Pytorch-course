import numpy as np
import matplotlib.pyplot as plt

### Sigmoid Function
sigmoid = lambda x: 1/(1+np.exp(-x))
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)

fig = plt.figure()
plt.plot(y, sigmoid(y), 'b', label = 'linspace(-10, 10, 10)')
plt.grid(linestyle='--')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Sigmoid Function')

plt.xticks([-4,-3,-2,-1,0,1,2,3,4])
plt.yticks([-2,-1,0,1,2])
plt.ylim(-2,2)
plt.xlim

plt.savefig('activation_graphs/sigmoid.png')
plt.show()

### TanH
tanh = lambda x: 2/(1+np.exp(-2*x)) -1
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)

plt.plot(y, tanh(y), 'b', label = 'linspace(-10, 10, 10)')
plt.grid(linestyle='--')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('TanH Function')
plt.xticks([-4,-3,-2,-1,0,1,2,3,4])
plt.yticks([-4, -3,-2,-1,0,1,2, 3, 4])
plt.ylim(-4,4)
plt.xlim(-4,4)
plt.savefig('activation_graphs/tanh.png')
plt.show()

#### ReLU
reul = lambda x: np.where(x>=0, x, 0)
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
plt.plot(y, reul(y), 'b', label = 'linspace(-10, 10, 10)')
plt.grid(linestyle='--')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('ReLU Function')
plt.xticks([-4,-3,-2,-1,0,1,2,3,4])
plt.yticks([-4, -3,-2,-1,0,1,2, 3, 4])
plt.ylim(-4,4)
plt.xlim(-4,4)
plt.savefig('activation_graphs/relu.png')
plt.show()

### Leaky ReLU
leaky_relu = lambda x: np.where(x>=0, x, 0.1*x)
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
plt.plot(y, leaky_relu(y), 'b', label = 'linspace(-10, 10, 10)')
plt.grid(linestyle='--')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Leaky ReLU Function')
plt.xticks([-4,-3,-2,-1,0,1,2,3,4])
plt.yticks([-4, -3,-2,-1,0,1,2, 3, 4])
plt.ylim(-4,4)
plt.xlim(-4,4)
plt.savefig('activation_graphs/leaky_relu.png')
plt.show()

### Binary Step Function
binary_step = lambda x: np.where(x>=0.0, 1.0, 0.0)
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
plt.plot(y, binary_step(y), 'b', label = 'linspace(-10, 10, 10)')
plt.grid(linestyle='--')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Binary Step Function')
plt.xticks([-4,-3,-2,-1,0,1,2,3,4])
plt.yticks([-2,-1,0,1,2])
plt.ylim(-2,2)
plt.xlim(-4,4)
plt.savefig('activation_graphs/binary_step.png')
plt.show()