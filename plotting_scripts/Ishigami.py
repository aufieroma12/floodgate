import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

import sys
sys.path.append('../config/')

from config import Hymod_inputs


FIG_DIR = '../figs/'
DATA_DIR = '../Hymod/data/analysis/{}/{}/'
if not os.path.exists(FIG_DIR):
    os.mkdir(FIG_DIR)

sample_sizes = [100, 250, 500, 1000, 5000, 10000, 50000]
X_labels = Hymod_inputs['labels']
d = len(X_labels)
gt = np.load(DATA_DIR.format('.','.') + 'ground_truth.npy')


# Plot formatting
lw = 2.75

SMALL_SIZE = 12
MEDIUM_SIZE = 20
BIG_SIZE = 24
BIGGER_SIZE = 30
HUGE_SIZE = 32

plt.rc('font', size=SMALL_SIZE)           # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIG_SIZE)       # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)   # fontsize of the figure title


### Figure 1 ###
fig, ax = plt.subplots(2, 5, figsize=(25,20), gridspec_kw={'height_ratios': [3, 1]})

floodgate = []
spf = []
spf_surrogate = []

# Floodgate, high-quality surrogate
for fp in os.listdir(DATA_DIR.format('floodgate', '100000')):
    floodgate.append(np.load(DATA_DIR.format('floodgate', '100000') + fp))
floodgate = np.array(floodgate)
floodgate_mean = np.mean(floodgate, axis=0)
floodgate_cov = np.mean((floodgate[:,:,:,0] < gt) * (floodgate[:,:,:,1] > gt), axis=0)

for i in range(d):
    ax[0,i].plot(sample_sizes, floodgate_mean[:,i,0], color='green', ls='-', lw=lw, label='Floodgate')
    ax[0,i].plot(sample_sizes, floodgate_mean[:,i,1], color='green', ls='-', lw=lw)
    ax[1,i].plot(sample_sizes, floodgate_cov[:,i], color='green', ls='-', lw=lw)

ax[0,2].plot([0], [20], color='grey', ls='--', lw=lw, label='Low-quality $f$')

# Non-surrogate SPF
for fp in os.listdir(DATA_DIR.format('spf', '.')):
    spf.append(np.load(DATA_DIR.format('spf', '.') + fp))
spf = np.array(spf)
spf_mean = np.mean(spf, axis=0)
spf_cov = np.mean((spf[:,:,:,0] < gt) * (spf[:,:,:,1] > gt), axis=0)

for i in range(d):
    ax[0,i].plot(sample_sizes, spf_mean[:,i,0], color='blue', ls='-', lw=lw, label='SPF $f^*$')
    ax[0,i].plot(sample_sizes, spf_mean[:,i,1], color='blue', ls='-', lw=lw)
    ax[1,i].plot(sample_sizes, spf_cov[:,i], color='blue', ls='-', lw=lw)

ax[0,2].plot([0], [20], color='grey', ls='-', lw=lw, label='High-quality $f$')

# Surrogate SPF, high-quality surrogate
for fp in os.listdir(DATA_DIR.format('spf_surrogate', '100000')):
    spf_surrogate.append(np.load(DATA_DIR.format('spf_surrogate', '100000') + fp))
spf_surrogate = np.array(spf_surrogate)
spf_surrogate_mean = np.mean(spf_surrogate, axis=0)
spf_surrogate_cov = np.mean((spf_surrogate[:,:,:,0] < gt) * (spf_surrogate[:,:,:,1] > gt), axis=0)

for i in range(d):
    ax[0,i].plot(sample_sizes, spf_surrogate_mean[:,i,0], color='orange', ls='-', lw=lw, label='SPF $f$')
    ax[0,i].plot(sample_sizes, spf_surrogate_mean[:,i,1], color='orange', ls='-', lw=lw)
    ax[1,i].plot(sample_sizes, spf_surrogate_cov[:,i], color='orange', ls='-', lw=lw)


floodgate = []
spf_surrogate = []

# Floodgate, low-quality surrogate
for fp in os.listdir(DATA_DIR.format('floodgate', '10000')):
    floodgate.append(np.load(DATA_DIR.format('floodgate', '10000') + fp))
floodgate = np.array(floodgate)
floodgate_mean = np.mean(floodgate, axis=0)
floodgate_cov = np.mean((floodgate[:,:,:,0] < gt) * (floodgate[:,:,:,1] > gt), axis=0)

for i in range(d):
    ax[0,i].plot(sample_sizes, floodgate_mean[:,i,0], color='green', ls='--', lw=lw)
    ax[0,i].plot(sample_sizes, floodgate_mean[:,i,1], color='green', ls='--', lw=lw)
    ax[1,i].plot(sample_sizes, floodgate_cov[:,i], color='green', ls='--', lw=lw)

# Surrogate SPF, low-quality surrogate
for fp in os.listdir(DATA_DIR.format('spf_surrogate', '10000')):
    spf_surrogate.append(np.load(DATA_DIR.format('spf_surrogate', '10000') + fp))
spf_surrogate = np.array(spf_surrogate)
spf_surrogate_mean = np.mean(spf_surrogate, axis=0)
spf_surrogate_cov = np.mean((spf_surrogate[:,:,:,0] < gt) * (spf_surrogate[:,:,:,1] > gt), axis=0)

for i in range(d):
    ax[0,i].plot(sample_sizes, spf_surrogate_mean[:,i,0], color='orange', ls='--', lw=lw)
    ax[0,i].plot(sample_sizes, spf_surrogate_mean[:,i,1], color='orange', ls='--', lw=lw)
    ax[1,i].plot(sample_sizes, spf_surrogate_cov[:,i], color='orange', ls='--', lw=lw)



x_ticks = [100, 1000, 10000, 100000]
x_tick_labs = ['100', '1000', '10000', '100000']
for i in range(d):
    ax[0,i].set(xscale='log', title=X_labels[i], ylim=(-.01,1))
    ax[1,i].set(xticks=x_ticks, xscale='log')
    ax[0,i].axhline(y=gt[i], color='red', ls=':', lw=lw, label='Target')
    ax[1,i].axhline(y=0.95, color='red', ls=':', lw=lw)
    ax[1,i].set_ylim((0,1.05))
    if i > 0:
        ax[0,i].set_yticks([])
        ax[1,i].set_yticks([])
    ax[0,i].set_xticks([])


ax[0,0].set(ylabel='Confidence Bounds for $S_j$')
ax[1,0].set(ylabel='Coverage')
ax[1,2].set_xlabel('Computational Budget $N$', fontsize=HUGE_SIZE)
ax[0,2].legend(loc='lower center', bbox_to_anchor=(0.5, -.6), ncol=3, fancybox=True)
ax[1,2].xaxis.labelpad = 15
fig.subplots_adjust(wspace=0.045, hspace=0.025)

# Save plots
plt.savefig(FIG_DIR + 'Hymod_bounds.png', bbox_inches="tight")





### Figure 2 ###
# Plot formatting
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIG_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


fig, ax = plt.subplots(2, 5, figsize=(25,10))

floodgate = []
panin = []

# Floodgate, high-quality surrogate
for fp in os.listdir(DATA_DIR.format('floodgate', 100000)):
    floodgate.append(np.load(DATA_DIR.format('floodgate', '100000') + fp))
floodgate = np.array(floodgate)
floodgate_mean = np.mean(floodgate[:,:,:,1] - floodgate[:,:,:,0], axis=0)
floodgate_cov = np.mean((floodgate[:,:,:,0] < gt) * (floodgate[:,:,:,1] > gt), axis=0)

for i in range(d):
    ax[0,i].plot(sample_sizes, floodgate_mean[:,i], color='green', ls='-', lw=lw, label='Floodgate')
    ax[1,i].plot(sample_sizes, floodgate_cov[:,i], color='green', ls='-', lw=lw)

ax[0,2].plot([0], [20], color='grey', ls='--', lw=lw, label='Low-quality $f$')

# Panin bound, high-quality surrogate
for fp in os.listdir(DATA_DIR.format('panin', 100000)):
    panin.append(np.load(DATA_DIR.format('panin', '100000') + fp))
panin = np.array(panin)
panin_mean = np.mean(panin[:,:,:,1] - panin[:,:,:,0], axis=0)
panin_cov = np.mean((panin[:,:,:,0] < gt) * (panin[:,:,:,1] > gt), axis=0)

for i in range(d):
    ax[0,i].plot(sample_sizes, panin_mean[:,i], color='purple', ls='-', lw=lw, label='Panin Bounds')
    ax[1,i].plot(sample_sizes, panin_cov[:,i], color='purple', ls='-', lw=lw)

ax[0,2].plot([0], [20], color='grey', ls='-', lw=lw, label='High-quality $f$')


floodgate = []
panin = []

# Floodgate, low-quality surrogate
for fp in os.listdir(DATA_DIR.format('floodgate', 10000)):
    floodgate.append(np.load(DATA_DIR.format('floodgate', '10000') + fp))
floodgate = np.array(floodgate)
floodgate_mean = np.mean(floodgate[:,:,:,1] - floodgate[:,:,:,0], axis=0)
floodgate_cov = np.mean((floodgate[:,:,:,0] < gt) * (floodgate[:,:,:,1] > gt), axis=0)

for i in range(d):
    ax[0,i].plot(sample_sizes, floodgate_mean[:,i], color='green', ls='--', lw=lw)
    ax[1,i].plot(sample_sizes, floodgate_cov[:,i], color='green', ls='--', lw=lw)

# Panin bound, low-quality surrogate
for fp in os.listdir(DATA_DIR.format('panin', 10000)):
    panin.append(np.load(DATA_DIR.format('panin', '10000') + fp))
panin = np.array(panin)
panin_mean = np.mean(panin[:,:,:,1] - panin[:,:,:,0], axis=0)
panin_cov = np.mean((panin[:,:,:,0] < gt) * (panin[:,:,:,1] > gt), axis=0)

for i in range(d):
    ax[0,i].plot(sample_sizes, panin_mean[:,i], color='purple', ls='--', lw=lw)
    ax[1,i].plot(sample_sizes, panin_cov[:,i], color='purple', ls='--', lw=lw)

ax[0,2].plot([0], [20], color='red', ls=':', lw=lw, label="Target")


# Plot formatting
for i in range(d):
    ax[0,i].set(xscale='log', title=X_labels[i])
    ax[1,i].set(xticks=x_ticks, xscale='log')   
    ax[1,i].axhline(y=0.95, color='red', ls=':', lw=lw)
    ax[0,i].set_ylim((0,1.01))
    ax[1,i].set_ylim((0,1.05))
    if i > 0:
        ax[0,i].set_yticks([])
        ax[1,i].set_yticks([])
    ax[0,i].set_xticks([])

ax[0,0].set(ylabel='Width of \nConfidence Intervals')
ax[1,0].set(ylabel='Coverage')
ax[1,2].set_xlabel('Computational Budget $N$', fontsize=HUGE_SIZE)
ax[0,2].legend(loc='lower center', bbox_to_anchor=(0.5, -1.7), ncol=3, fancybox=True)
ax[1,2].xaxis.labelpad = 15
fig.subplots_adjust(wspace=0.04, hspace=0.04)

# Save plots
plt.savefig(FIG_DIR + 'Hymod_panin_widths.png', bbox_inches="tight")

