import numpy as np
import matplotlib.pyplot as plt
 
x = np.linspace(-1, 1, 21)
y = np.linspace(-1, 1, 21)
x, y = np.meshgrid(x, y)
z = x * x + y * y - 1
 
fig = plt.figure(figsize=(12, 5))
 
ax1 = fig.add_subplot(121)
cntr = ax1.contourf(x, y, z)
ax1.clabel(cntr, colors='black')
ax1.set_aspect('equal')
 
ax2 = fig.add_subplot(122)
ax2.contourf(x, y, z)
cntr = ax2.contour(x, y, z)
ax2.clabel(cntr, colors='black')
ax2.set_aspect('equal')
 
plt.show()
