import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
 
x = np.linspace(-1, 1, 50)
y = np.linspace(-1, 1, 50)
x, y = np.meshgrid(x, y)
z3 = np.sqrt(x**2 + y**2)
z4 = np.abs(x) + np.abs(y)
 
h_ticks = np.linspace(-1, 1, 5)
v_ticks = np.linspace(0, 2, 5)
 
fig = plt.figure(figsize=(12, 5))
 
ax3 = fig.add_subplot(121, projection="3d")
ax3.set_xticks(h_ticks)
ax3.set_yticks(h_ticks)
ax3.set_zticks(v_ticks)
ax3.plot_surface(x, y, z3)
ax3.contour(x, y, z3, offset=0)
 
ax4 = fig.add_subplot(122, projection="3d")
ax4.set_xticks(h_ticks)
ax4.set_yticks(h_ticks)
ax4.set_zticks(v_ticks)
ax4.plot_surface(x, y, z4)
ax4.contour(x, y, z4, offset=0)
 
plt.show()
