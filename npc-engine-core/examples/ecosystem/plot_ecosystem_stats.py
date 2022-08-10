#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
d = np.genfromtxt('stats.txt', delimiter=',')
plt.plot(d)
plt.legend(['grass', 'herbivore', 'carnivore'])
plt.show()
