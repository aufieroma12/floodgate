from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import os

from src.analytical import S1, S2, S3, alpha, beta, gamma, analytical_mse

FIG_DIR = Path(__file__).parents[1] / "figs"
DATA_DIR = Path(__file__).parents[1] / "analytical_polynomial" / "data" / "analysis"

if not os.path.exists(FIG_DIR):
    os.mkdir(FIG_DIR)

d = 3
gt = np.array([S1, S2, S3])


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


### Figure 1 ###
fig, ax = plt.subplots(2, d, figsize=(15, 15), gridspec_kw={"height_ratios": [2, 1]})

floodgate = []
panin = []
spf_surrogate = []

noise_vals = ["0.5", "0.25", "0.1", "0.01", "0.001", "0.0005", "0.0"]

# Surrogate SPF
for s in noise_vals:
    data_dir = DATA_DIR / "spf_surrogate" / s
    spf_surrogate.append(
        [np.load(data_dir / fp) for fp in os.listdir(data_dir)]
    )
spf_surrogate = np.array(spf_surrogate)
spf_surrogate_mean = np.mean(spf_surrogate, axis=1)
spf_surrogate_cov = np.mean((spf_surrogate[:,:,:,0] < gt) * (spf_surrogate[:,:,:,1] > gt), axis=1)

# Floodgate
for s in noise_vals:
    data_dir = DATA_DIR / "floodgate" / s
    floodgate.append(
        [np.load(data_dir / fp) for fp in os.listdir(data_dir)]
    )

floodgate = np.array(floodgate)
floodgate_mean = np.mean(floodgate, axis=1)
floodgate_cov = np.mean((floodgate[:,:,:,0] < gt) * (floodgate[:,:,:,1] > gt), axis=1)


mse_vals = []
for n in noise_vals:
    noise = float(n)
    mse_vals.append(
        analytical_mse(alpha + noise, beta + 2 * noise, gamma - 0.5 * noise)
    )


for i in range(d):
    ax[0,i].plot(mse_vals, floodgate_mean[:,i,0], color="green", ls="-", lw=lw, label="Floodgate")
    ax[0,i].plot(mse_vals, floodgate_mean[:,i,1], color="green", ls="-", lw=lw)
    ax[1,i].plot(mse_vals, floodgate_cov[:,i], color="green", ls="-", lw=lw)


for i in range(d):
    ax[0,i].plot(mse_vals, spf_surrogate_mean[:,i,0], color="orange", ls="-", lw=lw, label="SPF $f$")
    ax[0,i].plot(mse_vals, spf_surrogate_mean[:,i,1], color="orange", ls="-", lw=lw)
    ax[1,i].plot(mse_vals, spf_surrogate_cov[:,i], color="orange", ls="-", lw=lw)


# Panin bounds
for s in noise_vals:
    data_dir = DATA_DIR / "panin" / s
    panin.append(
        [np.load(data_dir / fp) for fp in os.listdir(data_dir)]
    )
panin = np.array(panin)
panin_mean = np.mean(panin, axis=1)
panin_cov = np.mean((panin[:,:,:,0] < gt) * (panin[:,:,:,1] > gt), axis=1)

for i in range(d):
    ax[0,i].plot(mse_vals, panin_mean[:,i,0], color="purple", ls="-", lw=lw, label="Panin Bounds")
    ax[0,i].plot(mse_vals, panin_mean[:,i,1], color="purple", ls="-", lw=lw)
    ax[1,i].plot(mse_vals, panin_cov[:,i], color="purple", ls="-", lw=lw)


for i in range(d):
    ax[0,i].set(xscale="log", title=f"$X_{i+1}$", ylim=(-.01,1))
    ax[0,i].axhline(y=gt[i], color="red", ls=":", lw=lw, label="Target")
    ax[1,i].axhline(y=0.95, color="red", ls=":", lw=lw)
    ax[1,i].set_ylim((0,1.05))
    if i > 0:
        ax[0,i].set_yticks([])
        ax[1,i].set_yticks([])
    ax[0,i].set_xticks([])
    ax[0,i].invert_xaxis()
    ax[1,i].invert_xaxis()
    ax[1,i].set_xscale("log")
    ax[1,i].set_xticks([1e1, 1e-1, 1e-3, 1e-5])
    ax[1,i].set_xticklabels(["$10^1$", "$10^{-1}$", "$10^{-3}$", "$10^{-5}$"])


ax[0,0].set(ylabel="Confidence Bounds for $S_j$")
ax[1,0].set(ylabel="Coverage")
ax[1,1].set_xlabel("MSE(f)", fontsize=HUGE_SIZE)
ax[0,1].legend(loc="lower center", bbox_to_anchor=(0.5, -.85), ncol=2, fancybox=True)
ax[1,1].xaxis.labelpad = 15
fig.subplots_adjust(wspace=0.045, hspace=0.025)

# Save plots
plt.savefig(FIG_DIR / "analytical_function_bounds.png", bbox_inches="tight")
