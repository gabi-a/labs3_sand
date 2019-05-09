#!/usr/bin/env python3
import numpy as np
import sys
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print("Usage: plot_coeffs.py <dirname>\nwhere <dirname> contains the output from sandpile_findcoeffs.py")
dirname = sys.argv[1]

sizes = np.load(dirname+"/sizes.npy")
std = np.load(dirname+"/std.npy")

plt.figure()
plt.plot(sizes[:,0]*sizes[:,1], std,'x')
plt.xlabel("Area")
plt.ylabel("Standard deviation of amount of sand")
plt.title("Scale invariance of fluctuation")
plt.show()
