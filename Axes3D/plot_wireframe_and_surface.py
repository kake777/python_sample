import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
 
x = np.linspace(-1, 1, 20)
y = np.linspace(-1, 1, 20)
x, y = np.meshgrid(x, y)
z = x**2 + y**2
 
h_ticks = np.linspace(-1, 1, 5)
v_ticks = np.linspace(0, 2, 5)
 
fig = plt.figure(figsize=(12, 5))
 
ax1 = fig.add_subplot(121, projection="3d")
ax1.set_xticks(h_ticks)
ax1.set_yticks(h_ticks)
ax1.set_zticks(v_ticks)
ax1.plot_wireframe(x, y, z)
 
ax2 = fig.add_subplot(122, projection="3d")
ax2.set_xticks(h_ticks)
ax2.set_yticks(h_ticks)
ax2.set_zticks(v_ticks)
ax2.plot_surface(x, y, z)
 
plt.show()
