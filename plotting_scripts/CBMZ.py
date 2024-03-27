import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


from config import CBMZ_inputs


FIG_DIR = Path(__file__).parents[1] / "figs"
DATA_PATH = Path(__file__).parents[1] / "CBMZ" / "data" / "analysis"
if not os.path.exists(FIG_DIR):
    os.mkdir(FIG_DIR)

n_batches = [4, 7, 40, 79, 391, 625]
sample_sizes = [x * 128 for x in n_batches]
X_labels = CBMZ_inputs["labels"]
d = len(X_labels)
plot_subset = [2, 45, 10]


### Figure 3 ###
# Plot formatting
lw = 2.75

SMALL_SIZE = 12
MEDIUM_SIZE = 20
BIG_SIZE = 24
BIGGER_SIZE = 30
HUGE_SIZE = 32

plt.rc("font", size=BIG_SIZE)
plt.rc("axes", titlesize=BIGGER_SIZE, labelsize=BIGGER_SIZE)
plt.rc("xtick", labelsize=BIG_SIZE)
plt.rc("ytick", labelsize=BIG_SIZE)
plt.rc("legend", fontsize=BIG_SIZE)
plt.rc("figure", titlesize=BIGGER_SIZE)


fig, ax = plt.subplots(1, 3, figsize=(15, 6))

floodgate = np.load(DATA_PATH / "floodgate.npy")
spf = np.load(DATA_PATH / "spf.npy")
panin = np.load(DATA_PATH / "panin.npy")

for (i, idx) in enumerate(plot_subset):
    ax[i].plot(sample_sizes, floodgate[:,idx,0], color="green", ls="-", lw=lw, label="Floodgate")
    ax[i].plot(sample_sizes, floodgate[:,idx,1], color="green", ls="-", lw=lw)

    ax[i].plot(sample_sizes, spf[:,idx,0], color="orange", ls="-", lw=lw, label="SPF $f$")
    ax[i].plot(sample_sizes, spf[:,idx,1], color="orange", ls="-", lw=lw)

    ax[i].plot(sample_sizes, panin[:,idx,0], color="purple", ls="-", lw=lw, label="Panin Bounds")
    ax[i].plot(sample_sizes, panin[:,idx,1], color="purple", ls="-", lw=lw)


x_ticks = [1000, 10000, 100000]
for (i, idx) in enumerate(plot_subset):
    ax[i].set(xticks=x_ticks, xscale="log", title=X_labels[idx])
    ax[i].set_ylim((-0.02,1.02))

ax[1].set_yticks([])
ax[2].set_yticks([])

ax[0].set(ylabel="Confidence Bounds for $S_j$")
ax[1].set_xlabel("Computational Budget $N$")
ax[1].legend(loc="lower center", bbox_to_anchor=(0.5, -0.5), ncol=3, fancybox=True)
ax[1].xaxis.labelpad = 12
fig.subplots_adjust(wspace=0.035)

# Save plots
plt.savefig(FIG_DIR / "CBMZ_bounds.png", bbox_inches="tight")





### Extra Figure ###
# Plot formatting
lw = 2.5
ncols = 9

plt.rc("font", size=32)          # controls default text sizes
plt.rc("axes", titlesize=60)     # fontsize of the axes title
plt.rc("axes", labelsize=120)     # fontsize of the x and y labels
plt.rc("xtick", labelsize=52)    # fontsize of the tick labels
plt.rc("ytick", labelsize=52)    # fontsize of the tick labels
plt.rc("legend", fontsize=115)    # legend fontsize
plt.rc("figure", titlesize=90)   # fontsize of the figure title


fig, ax = plt.subplots(12, ncols, figsize=(100,120))

for (i, ax_) in enumerate(ax.ravel()[:d]):
    ax_.plot(sample_sizes, floodgate[:,i,0], color="green", ls="-", lw=lw, label="Floodgate")
    ax_.plot(sample_sizes, floodgate[:,i,1], color="green", ls="-", lw=lw)

    ax_.plot(sample_sizes, spf[:,i,0], color="orange", ls="-", lw=lw, label="SPF $f$")
    ax_.plot(sample_sizes, spf[:,i,1], color="orange", ls="-", lw=lw)

    ax_.plot(sample_sizes, panin[:,i,0], color="purple", ls="-", lw=lw, label="Panin Bounds")
    ax_.plot(sample_sizes, panin[:,i,1], color="purple", ls="-", lw=lw)



for (i, ax_) in enumerate(ax.ravel()[:d]):
    ax_.set(xscale="log", title=X_labels[i], ylim=(-0.02,1.02))
    if i % ncols > 0:
        ax_.set_yticks([])
    if i < d - ncols:
        ax_.set_xticks([])
    else:
        ax_.set_xticks(x_ticks)


ax[6,0].set(ylabel="\t\t   Confidence Bounds for $S_j$")
ax[-2,4].set_xlabel("Computational Budget $N$", fontsize=120, labelpad=550)
ax[-2,4].legend(loc="lower center", bbox_to_anchor=(0.5, -2), ncol=3, fancybox=True)
ax[7,5].xaxis.labelpad = 25

for i in range(2, ncols):
    ax[-1,i].axis("off")
fig.subplots_adjust(wspace=0.04, hspace=0.12)

# Save plots
plt.savefig(FIG_DIR / "CBMZ_full.png", bbox_inches="tight")

