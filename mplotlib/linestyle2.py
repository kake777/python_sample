import numpy as np
import matplotlib.pyplot as plt
 
fig = plt.figure()
ax = fig.add_subplot(111)
 
x = np.array([0, 1])
y = np.array([0, 1])
 
ax.plot(x, y, linestyle=(0, (1, 0)))
ax.plot(x, y+1, linestyle=(0, (0, 1)))
ax.plot(x, y+2, linestyle=(0, (1, 1)))
ax.plot(x, y+3, linestyle=(0, (5, 1)))
ax.plot(x, y+4, linestyle=(0, (5, 1, 1, 1)))
ax.plot(x, y+5, linestyle=(0, (5, 3, 1, 3, 1, 3)))
ax.plot(x, y+6, linestyle=(10, (5, 3, 1, 3, 1, 3)))
 
plt.show()
