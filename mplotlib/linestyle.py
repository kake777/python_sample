import numpy as np
import matplotlib.pyplot as plt
 
fig = plt.figure()
ax = fig.add_subplot(111)
 
x = np.array([0, 1])
y = np.array([0, 1])
 
ax.plot(x, y, linestyle="solid")
ax.plot(x, y+1, linestyle="dashed")
ax.plot(x, y+2, linestyle="dotted")
ax.plot(x, y+3, linestyle="dashdot")
 
plt.show()
