import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
 
x = [1, 1, 2, 2, 1, 1, 2, 2]
y = [1, 2, 1, 2, 1, 2, 1, 2]
z = [1, 1, 1, 1, 2, 2, 2, 2]
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim3d(0, 3)
ax.set_ylim3d(0, 3)
ax.set_zlim3d(0, 3)
ax.scatter(x, y, z)
 
plt.show()
