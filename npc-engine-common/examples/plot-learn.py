#!/usr/bin/env python3

import os
import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

print('Compiling')
os.system('cargo build --release --example learn')
Xs = []
for run in range(20):
	print(f'Running simulation {run}')
	os.system('target/release/examples/learn > output.csv')
	with open('output.csv') as f:
		Xs.append([float(x) for x in f.readlines()])
X = np.average(Xs, axis = 0)
avg_X = ndimage.uniform_filter1d(X, 100)
plt.plot(X)
plt.plot(avg_X)
plt.xlabel('epoch')
plt.ylabel('wood collected')
plt.show()