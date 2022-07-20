import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

import sys
sys.path.append('../config/')

from config import CBMZ_inputs


FIG_DIR = '../figs/'
DATA_PATH = '../CBMZ/data/analysis/{}.npy'
if not os.path.exists(FIG_DIR):
    os.mkdir(FIG_DIR)

n_batches = [4, 7, 40, 79, 391, 625]
sample_sizes = [x * 128 for x in n_batches]
X_labels = CBMZ_inputs['labels']
d = len(X_labels)
plot_subset = [2, 45, 10]


### Figure 3 ###
# Plot formatting
plt.rc('font', size=6)           # controls default text sizes
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=20)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=16)    # legend fontsize
plt.rc('figure', titlesize=24)   # fontsize of the figure title


fig, ax = plt.subplots(1, 3, figsize=(15,5))

floodgate = np.load(DATA_PATH.format('floodgate'))
spf = np.load(DATA_PATH.format('spf'))
panin = np.load(DATA_PATH.format('panin'))

for (i, idx) in enumerate(plot_subset):
    ax[i].plot(sample_sizes, floodgate[:,idx,0], color='green', ls='-', alpha=0.9, label='Floodgate')
    ax[i].plot(sample_sizes, floodgate[:,idx,1], color='green', ls='-', alpha=0.9)

    ax[i].plot(sample_sizes, spf[:,idx,0], color='orange', ls='-', alpha=0.9, label='SPF $f$')
    ax[i].plot(sample_sizes, spf[:,idx,1], color='orange', ls='-', alpha=0.9)

    ax[i].plot(sample_sizes, panin[:,idx,0], color='purple', ls='-', alpha=0.9, label='Panin Bounds')
    ax[i].plot(sample_sizes, panin[:,idx,1], color='purple', ls='-', alpha=0.9)


x_ticks = [100, 1000, 10000]
x_tick_labs = ['100', '1000', '10000']
for (i, idx) in enumerate(plot_subset):
    ax[i].set(xticks=x_ticks, xscale='log', title=X_labels[idx])
    ax[i].set_ylim((-0.02,1.02))

ax[0].set(ylabel='Confidence Bounds for $S_j$')
ax[1].set_xlabel('Computational Budget $N$')
ax[1].legend(loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=3, fancybox=True)
ax[1].xaxis.labelpad = 12

# Save plots
plt.savefig(FIG_DIR + 'CBMZ_bounds.png', bbox_inches="tight")





### Figure 4 ###
# Plot formatting
plt.rc('font', size=24)          # controls default text sizes
plt.rc('axes', titlesize=42)     # fontsize of the axes title
plt.rc('axes', labelsize=85)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=38)    # fontsize of the tick labels
plt.rc('ytick', labelsize=38)    # fontsize of the tick labels
plt.rc('legend', fontsize=55)    # legend fontsize
plt.rc('figure', titlesize=60)   # fontsize of the figure title


fig, ax = plt.subplots(13, 8, figsize=(85,130))

for (i, ax_) in enumerate(ax.ravel()[:d]):
    ax_.plot(sample_sizes, floodgate[:,i,0], color='green', ls='-', alpha=0.9, label='Floodgate')
    ax_.plot(sample_sizes, floodgate[:,i,1], color='green', ls='-', alpha=0.9)

    ax_.plot(sample_sizes, spf[:,i,0], color='orange', ls='-', alpha=0.9, label='SPF $f$')
    ax_.plot(sample_sizes, spf[:,i,1], color='orange', ls='-', alpha=0.9)

    ax_.plot(sample_sizes, panin[:,i,0], color='purple', ls='-', alpha=0.9, label='Panin Bounds')
    ax_.plot(sample_sizes, panin[:,i,1], color='purple', ls='-', alpha=0.9)


for (i, ax_) in enumerate(ax.ravel()[:d]):
    ax_.set(xticks=x_ticks, xscale='log', title=X_labels[i])
    ax_.set_ylim((-0.02,1.02))

ax[6,0].set(ylabel='Confidence Bounds for $S_j$')
ax[-1,3].set_xlabel('\t\t     Computational Budget $N$', fontsize=85)
ax[-1,3].legend(loc='lower center', bbox_to_anchor=(1.09, -0.65), ncol=3, fancybox=True)
ax[7,5].xaxis.labelpad = 25

for i in range(5,8):
    ax[-1,i].axis('off')

# Save plots
plt.savefig(FIG_DIR + 'CBMZ_full.png', bbox_inches="tight")

