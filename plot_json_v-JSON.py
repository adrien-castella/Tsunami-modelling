import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import writers

def animate(i):
    plt.clf()
    ax = sns.heatmap(data[i], vmin = minimum/2, vmax = maximum/2, center=0,
                         square = True, xticklabels = False, yticklabels = False,
                         cbar = True)

def save_animation(fig, name):
    anim = animation.FuncAnimation(fig, animate, interval = 20, frames = len(data))

    Writer = writers['ffmpeg']
    writer = Writer(fps=50, metadata={'artist': 'Me'}, bitrate=1000)

    anim.save('Tsunami Simulation '+name+'.mp4',writer)

name = input("Enter the name: ")
maximum = float(input("Enter maximum: "))
minimum = -maximum

data = 0
with open(name+"_grid.json", "r") as read_file:
	data = json.load(read_file)

data = np.array(data)

fig = plt.figure()
save_animation(fig, name)