#!/usr/bin/env python3

"""
 SPDX-License-Identifier: Apache-2.0 OR MIT
 © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
"""

import sys
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
os.system('rm -f output.csv')
X = np.average(Xs, axis = 0)
avg_X = ndimage.uniform_filter1d(X, 100)
plot_to_file = len(sys.argv) > 1
if plot_to_file:
	plt.rcParams["figure.figsize"] = (3.3, 1.6)
plt.plot(X)
plt.plot(avg_X)
plt.xlabel('epoch')
plt.ylabel('wood collected')
if plot_to_file:
	plt.savefig(sys.argv[1], bbox_inches='tight')
else:
	plt.show()