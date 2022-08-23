#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
d = np.genfromtxt('stats.txt', delimiter=',')

fig, (ax1, ax2) = plt.subplots(2)

ax1.plot(d[:,[1,2]])
ax1.set_ylabel('animal count')
ax1.legend(['herbivore', 'carnivore'])

ax2.plot(d[:,[0,3]])
ax2.set_ylabel('grass and visit count')
ax2.legend(['grass', 'visits'])

plt.show()
