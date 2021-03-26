import numpy as np
import matplotlib.pyplot as plt
 
x = np.linspace(-1, 1, 21)
y = np.linspace(-1, 1, 21)
x, y = np.meshgrid(x, y)
z1 = (x + 0.5)**2 + y * y - 1
z2 = (x - 0.5)**2 + y * y - 1
 
fig = plt.figure(figsize=(12, 5))
 
ax1 = fig.add_subplot(131)
ax1.contourf(x, y, z1, cmap='autumn', alpha=0.5)
ax1.set_aspect('equal')
 
ax2 = fig.add_subplot(132)
ax2.contourf(x, y, z2, cmap='winter', alpha=0.5)
ax2.set_aspect('equal')
 
ax3 = fig.add_subplot(133)
ax3.contourf(x, y, z1, cmap='autumn', alpha=0.5)
ax3.contourf(x, y, z2, cmap='winter', alpha=0.5)
ax3.set_aspect('equal')
 
plt.show()
