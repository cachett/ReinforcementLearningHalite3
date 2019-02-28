
import matplotlib.pyplot as plt
import scipy
import scipy.signal
import numpy as np
import math

# Design the Gaussian filter
def gaussian_filter_1d(sigma):
    # sigma: the parameter sigma in the Gaussian kernel (unit: pixel)
    #
    # return: a 1D array for the Gaussian kernel

    # The filter radius is 3.5 times sigma
    rad = int(math.ceil(3.5 * sigma))
    sz = 2 * rad + 1
    h = np.zeros((sz,))
    for i in range(-rad, rad + 1):
        h[ i + rad] = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(i * i)/(2 * sigma * sigma))
    h = h / np.sum(h)
    return [h]

file = open("progress.txt", "r")
file = file.readlines()

smoothing = True
filter = gaussian_filter_1d(4)

values = []
for line in file:
    line = line.split(' ')
    if len(line) > 1:
        values.append(float(line[-1]))

total_halite = [elt for index, elt in enumerate(values) if index%4==0]
total_reward = [elt for index, elt in enumerate(values) if index%4==1]
total_ships = [elt for index, elt in enumerate(values) if index%4==2]
total_reward_per_iteration = [elt for index, elt in enumerate(values) if index%4==3]
x = [i for i in range(len(total_halite))]

if smoothing:
    total_halite = scipy.signal.convolve2d([total_halite], filter, mode='valid')[0]
    total_reward = scipy.signal.convolve2d([total_reward], filter, mode='valid')[0]
    total_ships = scipy.signal.convolve2d([total_ships], filter, mode='valid')[0]
    total_reward_per_iteration = scipy.signal.convolve2d([total_reward_per_iteration], filter, mode='valid')[0]
    x = [i for i in range(len(total_halite))]


f, axarr = plt.subplots(4, sharex=True)
axarr[0].plot(x, total_halite)
axarr[0].set_title('Total halite')
axarr[1].plot(x, total_reward)
axarr[1].set_title('Total reward')
axarr[2].plot(x, total_ships)
axarr[2].set_title('Total ships')
axarr[3].plot(x, total_reward_per_iteration)
axarr[3].set_title('Reward per it')

plt.show()
