import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
 
t = np.linspace(0, 8 * np.pi, num=200)
x = t * np.cos(t)
y = t * np.sin(t)
z = t
 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
 
ax.plot(x, y, z)
 
plt.show()
