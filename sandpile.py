#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from util import multipage

do_animation = False

size_x = 10
size_y = 10
n_drops = 100000

table = np.zeros((n_drops, size_x, size_y))
avalanche_size = np.zeros(n_drops, dtype=int)
avalanche_area = np.zeros(n_drops, dtype=int)
avalanche_time = np.zeros(n_drops, dtype=int)
avalanche_radius = np.zeros(n_drops, dtype=int)

drop_points = np.random.randint(0, size_x, (n_drops,2))
# drop_points = np.ones((n_drops,2),dtype=int)
# drop_points = np.zeros((n_drops, 2), dtype=int)

for t in range(1,n_drops):

    print(f"{100*t/n_drops:.0f}%",end="\r")

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

def fit_and_plot(data, xlabel="", cutoff=-1):
    x = data[0]
    y = data[1]

    cutoff = min(len(x)-1, cutoff)

    popt, pcov = curve_fit(lambda x,a,b: a*x+b, np.log(x[:cutoff]), np.log(y[:cutoff]))
    # popt2, pcov2 = curve_fit(lambda x,a,b: a*x+b, np.log(x[cutoff:]), np.log(y[cutoff:]))
    fig, ax = plt.subplots()
    plt.loglog(x, y, 'x')
    xfine = np.linspace(x[0],x[cutoff])
    # xfine2 = np.linspace(x[cutoff],x[-1])
    plt.loglog(xfine, np.exp(popt[1])*np.power(xfine, popt[0]))
    # plt.loglog(xfine2, np.exp(popt2[1])*np.power(xfine2, popt2[0]))
    ax.text(0.4,0.9, "$y=%.2fx^{%.3f}$"%(np.exp(popt[1]), popt[0]),transform=ax.transAxes)
    # ax.text(0.4,0.8, "$y=%.2fx^{%.3f}$"%(np.exp(popt2[1]), popt2[0]),transform=ax.transAxes)
    ax.text(0.74,0.85, f"size x = {size_x}\nsize y = {size_y}\n#drops = {n_drops}",bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 5},transform=ax.transAxes)
    plt.ylabel("frequency")
    plt.xlabel(xlabel)


fit_and_plot(avalanche_size_freqs, "size", cutoff=min(size_x, size_y))
plt.title("avalanche size")

fit_and_plot(avalanche_area_freqs, "area", cutoff=min(size_x, size_y))
plt.title("avalanche area")

fit_and_plot(avalanche_time_freqs, "time", cutoff=min(size_x, size_y))
plt.title("avalanche time")

fit_and_plot(avalanche_radius_freqs, "radius", cutoff=min(size_x, size_y))
plt.title("avalanche radius")

if do_animation:
    from matplotlib import animation
    fig = plt.figure()
    ims = []
    for i in range(n_drops):
        im = plt.imshow(table[i,:,:], animated=True, cmap='terrain', vmin=0, vmax=4)
        ims.append([im])

    anim = animation.ArtistAnimation(fig, ims, interval=100, blit=True)

multipage(f"{n_drops}_{size_x}x{size_y}.pdf")
# plt.show()
