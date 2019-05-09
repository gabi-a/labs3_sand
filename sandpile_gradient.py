#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from util import multipage
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-x","--sizex",help="x sidelength of grid",type=int,default=25)
parser.add_argument("-y","--sizey",help="y sidelength of grid",type=int,default=25)
parser.add_argument("-n","--ndrops",help="number of sand grains to drop",type=int,default=10000)
parser.add_argument("-t","--toppleheight",help="topple height threshold",type=int,default=4)
parser.add_argument("-d","--dropamount",help="number of grains to drop per drop",type=int,default=1)
parser.add_argument("-a","--animate",help="create an animation of the sand",type=bool,default=False)
args = parser.parse_args()

size_x = args.sizex
size_y = args.sizey
n_drops = args.ndrops
do_animation = args.animate

table = np.zeros((n_drops, size_x, size_y))
avalanche_size = np.zeros(n_drops, dtype=int)
avalanche_area = np.zeros(n_drops, dtype=int)
avalanche_time = np.zeros(n_drops, dtype=int)
avalanche_radius = np.zeros(n_drops, dtype=int)

# drop_points = np.random.randint(0, size_x, (n_drops,2))
drop_amount = args.dropamount
topple_height = args.toppleheight
drop_points = int(size_x/2) * np.ones((n_drops,2),dtype=int)
# drop_points = np.zeros((n_drops, 2), dtype=int)

table[0,:,:] = topple_height
drop_amount = 0

for t in range(1,n_drops):

    print(f"{100*t/n_drops:.0f}%",end="\r")

    table[t,:,:] = table[t-1,:,:]
    table[t,drop_points[t,0],drop_points[t,1]] += drop_amount

    toppled = np.zeros((size_x,size_y),dtype=bool)
    while(True):
        delta = np.zeros((table.shape[1],table.shape[2]))

        # Topple up
        mask_up = table[t,:,:].copy()
        mask_down = table[t,:,:].copy()

        mask_up[:,0] = -np.inf
        mask_down[:,-1] = -np.inf

        delta_up = np.where(table[t,1:,:]-table[t,:-1,:]>=topple_height)
        num_to_topple = delta_up[0].shape[0]

        if num_to_topple == 0:
            break

        gain_x = delta_up[1]
        gain_y = delta_up[0]
        lose_x = gain_x
        lose_y = gain_y + 1

        delta[gain_y, gain_x] += 1
        delta[lose_y, lose_x] -= 1

        table[t,:,:] += delta
        delta[:,:] = 0

        # Topple down
        # delta_down = np.where((table[t,:-1,:]-table[t,1:,:])>=topple_height)
        # num_to_topple = delta_down[0].shape[0]
        #
        # if num_to_topple == 0:
        #     break
        #
        # lose_x = delta_down[0]
        # lose_y = delta_down[1]
        # gain_x = lose_x
        # gain_y = lose_y + 1
        #
        # delta[gain_y, gain_x] += 1
        # delta[lose_y, lose_x] -= 1
        #
        # table[t,:,:] += delta
        # delta[:,:] = 0
        #
        # # Topple left
        # delta_left = np.where(table[t,:,1:]-table[t,:,:-1]>=topple_height)
        # num_to_topple = delta_left[0].shape[0]
        #
        # if num_to_topple == 0:
        #     break
        #
        # gain_x = delta_left[1]
        # gain_y = delta_left[0]
        # lose_x = gain_x + 1
        # lose_y = gain_y
        #
        # delta[gain_y, gain_x] += 1
        # delta[lose_y, lose_x] -= 1
        #
        # table[t,:,:] += delta
        # delta[:,:] = 0
        #
        # # Topple right
        # delta_right = np.where(table[t,:,:-1]-table[t,:,1:]>=topple_height)
        # num_to_topple = delta_right[0].shape[0]
        #
        # if num_to_topple == 0:
        #     break
        #
        # lose_x = delta_right[1]
        # lose_y = delta_right[0]
        # gain_x = lose_x + 1
        # gain_y = lose_y
        #
        # delta[gain_y, gain_x] += 1
        # delta[lose_y, lose_x] -= 1
        #
        # table[t,:,:] += delta
        # delta[:,:] = 0

        # to_topple = table[t,:,:] >= topple_height
        # if np.any(to_topple) == 0:
        #     break
        #
        #
        # num_to_topple = np.sum(to_topple)
        #
        # toppled[to_topple] = True
        # avalanche_time[t] += 1
        # avalanche_size[t] += num_to_topple
        #
        # topple_locs = np.where(to_topple)
        #
        # table[t][topple_locs] -= 4
        #
        # mask = np.zeros((table.shape[1],table.shape[2]))
        # new = mask.copy()
        # mask[topple_locs] = 1
        #
        # new[:-1,:] += mask[1:,:]  # up
        # new[1:,:]  += mask[:-1,:] # down
        # new[:,:-1] += mask[:,1:]  # left
        # new[:,1:]  += mask[:,:-1] # left
        #
        # xy = np.where(new)
        # x = xy[0]
        # y = xy[1]
        # u = np.max(np.abs(x-drop_points[t,0]) + np.abs(y-drop_points[t,1]))
        # avalanche_radius[t] = np.max((avalanche_radius[t], u))
        #
        # table[t,:,:] += new


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
guess_height = size_x * size_y * (topple_height - 2)
plt.plot(range(total_sand.shape[0]), guess_height * np.ones(total_sand.shape[0]))
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

    if len(x) == 0 or len(y) == 0:
        print(f"No data for {xlabel} - try running a longer simulation!")
        return

    cutoff = min(len(x)-1, cutoff)

    popt, pcov = curve_fit(lambda x,a,b: a*x+b, np.log(x[:cutoff]), np.log(y[:cutoff]))
    fig, ax = plt.subplots()
    plt.loglog(x, y, 'x')
    xfine = np.linspace(x[0],x[cutoff])
    plt.loglog(xfine, np.exp(popt[1])*np.power(xfine, popt[0]))
    ax.text(0.4,0.9, "$y=%.2fx^{%.3f}$"%(np.exp(popt[1]), popt[0]),transform=ax.transAxes)
    ax.text(0.73,0.85, f"size x = {size_x}\nsize y = {size_y}\n#drops = {n_drops}",bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 5},transform=ax.transAxes)
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

fig, ax = plt.subplots()
plt.imshow(table[-1,:,:], cmap='terrain', vmin=0, vmax=topple_height)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('sand at final timestep')
ax.text(0.65,0.85, f"size x = {size_x}\nsize y = {size_y}\n#drops = {n_drops}",bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 5},transform=ax.transAxes)

fname = f"{n_drops}_{size_x}x{size_y}_T{topple_height}_D{drop_amount}"
print("Saving plots to "+fname+".pdf")
multipage(fname+".pdf")

if do_animation:
    print("Animating - this may take a while...")
    print()
    plt.close("all")
    from matplotlib import animation
    fig = plt.figure()
    ims = []
    for i in range(n_drops):
        print(f"{100*i/n_drops:.0f}% rendered", end="\r")
        im = plt.imshow(table[i,:,:], animated=True, cmap='terrain', vmin=0, vmax=4)
        ims.append([im])

    anim = animation.ArtistAnimation(fig, ims, interval=100, blit=True)
    plt.colorbar()

    plt.show()
