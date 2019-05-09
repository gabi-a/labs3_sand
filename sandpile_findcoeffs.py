#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from util import multipage
import argparse
import os
from sandpileClass import Sandpile

parser = argparse.ArgumentParser()
parser.add_argument("-x","--maxlength",help="max sidelength of grid",type=int,default=25)
parser.add_argument("-N","--nruns",help="number of runs to do",type=int,default=10)
parser.add_argument("-n","--ndrops",help="number of sand grains to drop",type=int,default=10000)
parser.add_argument("-t","--toppleheight",help="topple height threshold",type=int,default=4)
parser.add_argument("-d","--dropamount",help="number of grains to drop per drop",type=int,default=1)
parser.add_argument("-a","--animate",help="create an animation of the sand",type=bool,default=False)
parser.add_argument("-h0","--initialheight",help="initial height of sand",type=int,default=0)
args = parser.parse_args()

max_length = args.maxlength
N = args.nruns
n_drops = args.ndrops
do_animation = args.animate
initial_height = args.initialheight

drop_amount = args.dropamount
topple_height = args.toppleheight

def find_linear_region(data, natural, label=""):

    x = data[0]
    y = data[1]

    if x.shape[0] < 2:
        print(f"Not enough data to analyse {label}")
        return 0

    if x.shape != y.shape:
        print("Oh no! Arrays mismatched")
        return 0

    # Step 1: Select the region that is definetly linear
    cutoff = int(np.round(natural / 2))
    if cutoff >= x.shape[0] or cutoff < 2:
        print(f"Not enough data to analyse {label}")
        return 0

    # Step 2: Fit a line to this region
    popt, pcov = curve_fit(lambda x,a,b: a*x+b, np.log(x[:cutoff]), np.log(y[:cutoff]))

    # Step 3: Find the max distance between the line and the data in the linear region
    maxd = 1.01 * np.max(np.abs((popt[0] * np.log(x[:cutoff]) + popt[1]) - np.log(y[:cutoff])))

    # Step 3: Find the distance between data and the line
    d = np.abs((popt[0] * np.log(x) + popt[1]) - np.log(y))

    # Step 4: Find the x value where the data first deviates from the line
    # print(d.shape)
    # print(x.shape)
    # print(maxd)
    m = np.where(d > maxd)[0]
    if len(m) == 0:
        print(f"Not enough data to analyse {label}")
        return 0
    cutoff = x[m[0]]

    return cutoff / natural

sizes = np.random.choice(np.arange(1, max_length), (N,2))

size_cutoff = np.zeros(N)
time_cutoff = np.zeros(N)
radius_cutoff = np.zeros(N)
area_cutoff = np.zeros(N)

dirname = f"{max_length}_{N}_{n_drops}"
if not os.path.exists(dirname):
    os.mkdir(dirname)
else:
    print("Directory already exists!")
    input("Press any key to continue...")

np.save(dirname+"/sizes", sizes)

for i in range(N):

    print(f"{i+1}/{N}          ")

    drop_points_x = np.random.randint(0, sizes[i,0], n_drops)
    drop_points_y = np.random.randint(0, sizes[i,1], n_drops)

    if initial_height == -1:
        initial_height = (topple_height - 2)

    sandpile = Sandpile(sizes[i,0], sizes[i,1], n_drops, initial_height, drop_points_x, drop_points_y, drop_amount, topple_height)
    try:
        sandpile.run()

        size_cutoff[i] = find_linear_region(sandpile.avalanche_size_freqs, sizes[i,0]*sizes[i,1], "size")
        time_cutoff[i] = find_linear_region(sandpile.avalanche_time_freqs, min(sizes[i,0],sizes[i,1]), "time")
        radius_cutoff[i] = find_linear_region(sandpile.avalanche_radius_freqs, min(sizes[i,0],sizes[i,1]), "radius")
        area_cutoff[i] = find_linear_region(sandpile.avalanche_area_freqs, sizes[i,0]*sizes[i,1], "area")

        np.save(dirname+"/size", size_cutoff)
        np.save(dirname+"/time", time_cutoff)
        np.save(dirname+"/radius", radius_cutoff)
        np.save(dirname+"/area", area_cutoff)
    except Exception as e:
        print(e)
