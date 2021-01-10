import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import writers

def animate(i):
    plt.clf()
    ax = sns.heatmap(data[i], vmin = minimum/2, vmax = maximum/2,
                         square = True, xticklabels = False, yticklabels = False,
                         cbar = False)

def save_animation(fig, name):
    anim = animation.FuncAnimation(fig, animate, interval = 100, frames = len(data))

    Writer = writers['ffmpeg']
    writer = Writer(fps=10, metadata={'artist': 'Me'}, bitrate=1000)

    anim.save(name+'_channel_vid.mp4', writer)

name = input("Enter the name: ")

data = 0
with open(name+"_data_py.json", "r") as read_file:
    data = json.load(read_file)

data = np.array(data)
minimum = np.amin(data)
maximum = np.amax(data)

fig = plt.figure()
save_animation(fig, name)