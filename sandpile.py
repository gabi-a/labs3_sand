#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

do_animation = False

size_x = 25
size_y = 25
n_drops = 10000

table = np.zeros((n_drops, size_x, size_y))
avalanche_size = np.zeros(n_drops, dtype=int)
avalanche_area = np.zeros(n_drops, dtype=int)
avalanche_time = np.zeros(n_drops, dtype=int)
avalanche_radius = np.zeros(n_drops, dtype=int)

drop_points = np.random.randint(0, size_x, (n_drops,2))
# drop_points = np.ones((n_drops,2),dtype=int)
# drop_points = np.zeros((n_drops, 2), dtype=int)

for t in range(1,n_drops):

    # print(f"{t}/{n_drops}",end="\r")

    table[t,:,:] = table[t-1,:,:]
    table[t,drop_points[t,0],drop_points[t,1]] += 1
    toppled = np.zeros((size_x,size_y),dtype=bool)
    while(True):

        to_topple = table[t,:,:] >= 4
        if np.any(to_topple) == 0:
            break

        num_to_topple = np.sum(to_topple)

        toppled[to_topple] = True
        avalanche_time[t] += 1
        avalanche_size[t] += num_to_topple

        topple_locs = np.where(to_topple)

        table[t][topple_locs] -= 4

        mask = np.zeros((table.shape[1],table.shape[2]))
        new = mask.copy()
        mask[topple_locs] = 1

        new[:-1,:] += mask[1:,:] # up
        new[1:,:] += mask[:-1,:] # down
        new[:,:-1] += mask[:,1:] # left
        new[:,1:] += mask[:,:-1] # left

        xy = np.where(new)
        x = xy[0]
        y = xy[1]
        u = np.max(np.abs(x-drop_points[t,0]) + np.abs(y-drop_points[t,1]))
        avalanche_radius[t] = np.max((avalanche_radius[t], u))

        table[t,:,:] += new

    avalanche_area[t] = np.sum(toppled)

total_sand = np.sum(table, (1,2))

avalanche_size = 4*avalanche_size[np.where(avalanche_size != 0)]
avalanche_area = avalanche_area[np.where(avalanche_area != 0)]
avalanche_time = avalanche_time[np.where(avalanche_time != 0)]
avalanche_radius = avalanche_radius[np.where(avalanche_radius != 0)]

avalanche_size_freqs = np.unique(avalanche_size, return_counts=True)
avalanche_area_freqs = np.unique(avalanche_area, return_counts=True)
avalanche_time_freqs = np.unique(avalanche_time, return_counts=True)
avalanche_radius_freqs = np.unique(avalanche_radius, return_counts=True)

plt.figure()
plt.plot(total_sand)
plt.xlabel("t")
plt.ylabel("amount")
plt.title("Total sand on table")

# plt.figure()
# x = np.linspace(0,10)
# plt.loglog(x, power_law(x, 1, 2, 0))
# plt.show()
# quit()

def fit_and_plot(data, cutoff=-1):
    x = data[0]
    y = data[1]

    popt, pcov = curve_fit(lambda x,a,b: a*x+b, np.log(x[:cutoff]), np.log(y[:cutoff]))
    plt.loglog(x, y, 'x')
    xfine = np.linspace(x[0],x[cutoff])
    plt.loglog(xfine, np.exp(popt[1])*np.power(xfine, popt[0]))

plt.figure()
fit_and_plot(avalanche_size_freqs, cutoff=80)
plt.title("avalanche size")

plt.figure()
fit_and_plot(avalanche_area_freqs, cutoff=50)
plt.title("avalanche area")

plt.figure()
fit_and_plot(avalanche_time_freqs)
plt.title("avalanche time")

plt.figure()
fit_and_plot(avalanche_radius_freqs)
plt.title("avalanche radius")

if do_animation:
    from matplotlib import animation
    fig = plt.figure()
    ims = []
    for i in range(n_drops):
        im = plt.imshow(table[i,:,:], animated=True, cmap='terrain', vmin=0, vmax=4)
        ims.append([im])

    anim = animation.ArtistAnimation(fig, ims, interval=100, blit=True)

plt.show()
