import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1, 1, 100)[:, np.newaxis]          # shape (100, 1)
noise = np.random.normal(0, 0.1, size=x.shape)
y = np.power(x, 2) + noise 

plt.figure()
plt.subplot(1,2,1)
plt.scatter(x,y)

plt.subplot(1,2,2)
plt.plot(x,y)
plt.show()