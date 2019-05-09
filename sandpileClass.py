import numpy as np

class Sandpile:

    def __init__(self, size_x, size_y, n_drops, initial_height, drop_points_x, drop_points_y, drop_amount, topple_height):
        self.size_x = size_x
        self.size_y = size_y
        self.n_drops = n_drops
        self.initial_height = initial_height
        self.drop_points_x = drop_points_x
        self.drop_points_y = drop_points_y
        self.drop_amount = drop_amount
        self.topple_height = topple_height

        self.table = np.zeros((n_drops, size_x, size_y))
        self.avalanche_size = np.zeros(n_drops, dtype=int)
        self.avalanche_area = np.zeros(n_drops, dtype=int)
        self.avalanche_time = np.zeros(n_drops, dtype=int)
        self.avalanche_radius = np.zeros(n_drops, dtype=int)

        self.total_sand = None
        self.avalanche_size_freqs = None
        self.avalanche_area_freqs = None
        self.avalanche_time_freqs = None
        self.avalanche_radius_freqs = None

    def run(self):

        self.table[0,:,:] = self.initial_height

        for t in range(1,self.n_drops):

            print(f"{100*t/self.n_drops:.0f}%",end="\r")

            self.table[t,:,:] = self.table[t-1,:,:]
            self.table[t,self.drop_points_x[t],self.drop_points_y[t]] += self.drop_amount

            toppled = np.zeros((self.size_x,self.size_y),dtype=bool)
            while(True):

                to_topple = self.table[t,:,:] >= self.topple_height
                if np.any(to_topple) == 0:
                    break

                num_to_topple = np.sum(to_topple)

                toppled[to_topple] = True
                self.avalanche_time[t] += 1
                self.avalanche_size[t] += num_to_topple

                topple_locs = np.where(to_topple)

                self.table[t][topple_locs] -= 4

                mask = np.zeros((self.table.shape[1],self.table.shape[2]))
                new = mask.copy()
                mask[topple_locs] = 1

                new[:-1,:] += mask[1:,:]  # up
                new[1:,:]  += mask[:-1,:] # down
                new[:,:-1] += mask[:,1:]  # left
                new[:,1:]  += mask[:,:-1] # left

                xy = np.where(new)
                x = xy[0]
                y = xy[1]
                try:
                    u = np.max(np.abs(x-self.drop_points_x[t]) + np.abs(y-self.drop_points_y[t]))
                    self.avalanche_radius[t] = np.max((self.avalanche_radius[t], u))
                except Exception as e:
                    print(e)
                    
                self.table[t,:,:] += new

            self.avalanche_area[t] = np.sum(toppled)

        self.total_sand = np.sum(self.table, (1,2))
        self.avalanche_size = 4*self.avalanche_size[np.where(self.avalanche_size != 0)]
        self.avalanche_area = self.avalanche_area[np.where(self.avalanche_area != 0)]
        self.avalanche_time = self.avalanche_time[np.where(self.avalanche_time != 0)]
        self.avalanche_radius = self.avalanche_radius[np.where(self.avalanche_radius != 0)]

        self.avalanche_size_freqs = np.unique(self.avalanche_size, return_counts=True)
        self.avalanche_area_freqs = np.unique(self.avalanche_area, return_counts=True)
        self.avalanche_time_freqs = np.unique(self.avalanche_time, return_counts=True)
        self.avalanche_radius_freqs = np.unique(self.avalanche_radius, return_counts=True)
