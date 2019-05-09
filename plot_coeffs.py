#!/usr/bin/env python3
import numpy as np
import sys
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print("Usage: plot_coeffs.py <dirname>\nwhere <dirname> contains the output from sandpile_findcoeffs.py")
dirname = sys.argv[1]

size_cutoff = np.load(dirname+"/size.npy")
time_cutoff = np.load(dirname+"/time.npy")
radius_cutoff = np.load(dirname+"/radius.npy")
area_cutoff = np.load(dirname+"/area.npy")
sizes = np.load(dirname+"/sizes.npy")

min_sizes = np.min(sizes, axis=1)
areas = sizes[:,0] * sizes[:,1]
# size_cutoff = size_cutoff[np.where((size_cutoff != 0))]
# time_cutoff = time_cutoff[np.where((time_cutoff != 0) & (time_cutoff < 1))]
# radius_cutoff = radius_cutoff[np.where((radius_cutoff != 0) & (radius_cutoff < 1))]
# area_cutoff = area_cutoff[np.where((area_cutoff != 0) & (area_cutoff < 1))]

plt.figure()
plt.plot(min_sizes, size_cutoff, 'x')
plt.title("size_cutoff")
plt.xlabel("min length")

plt.figure()
plt.plot(areas, time_cutoff, 'x')
plt.title("time_cutoff")
plt.xlabel("area")

plt.figure()
plt.plot(min_sizes, radius_cutoff, 'x')
plt.title("radius_cutoff")
plt.xlabel("min length")

plt.figure()
plt.plot(areas, area_cutoff, 'x')
plt.title("area_cutoff")
plt.xlabel("area")

plt.show()
