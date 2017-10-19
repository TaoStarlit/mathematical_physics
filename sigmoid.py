# import math

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

# x = range(-5, 5, 0.1)
# print("range x type:", type(x))  #<type 'list'>
x = np.arange(-5, 5, 0.5)
print("np.arange x type:", type(x))
sigmoid = 1 / (1 + np.exp(-x))
hyperbolic_tangent = (1 - np.exp(-x)) / (1 + np.exp(-x))

print(sigmoid)

plt.figure("softmax normalization curves")

plt.subplot(2, 2, 1)
plt.title('hyperbolic_tangent curve')
plt.plot(x, hyperbolic_tangent, c="blue")

plt.subplot(2, 2, 2)
plt.title('sigmoid curve')
plt.plot(x, sigmoid, label='sigmoid curve', c="red")

cmap = ListedColormap(['#FF0000', '#00FF00'])#generate a cmap object
cmap = plt.cm.get_cmap("rainbow")#get a cmap instance
for i in range(2, 6, 1):
    sigmoid = 1 / (1 + np.exp(-i * x))

    plt.subplot(2, 4, i + 3)
    plt.title('sigmoid curve' + str(i))
    c = cmap(float(i) / 4)
    plt.plot(x, sigmoid, c=c, alpha=.4)

    
plt.show()
