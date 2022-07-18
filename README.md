# Floodgate for Sensitivity Analysis
This repository contains the source code accompanying our paper: Aufiero, Massimo and Janson, Lucas. "Surrogate-based global sensitivity analysis with statistical guarantees via floodgate." arXiv preprint ___ (2022).

## Instructions for Reproducing Results
The scripts in this repository were run on the FASRC Cannon cluster supported by the FAS Division of Science Research Computing Group at Harvard University. We have included `.sbatch` scripts and corresponding bash scripts for submitting parallel jobs that are compatible with any Linux-based cluster using the Slurm workload manager. While it is possible to run the Python scripts directly, serial computation time would be on the order of 10,000 hours.

### Setting up
After cloning this repository onto your local machine or compute cluster, change the `BASE_DIR` variable in `config/config.py` to the appropriate file path to this directory, and run 
```
export BASE_DIR=/path/to/floodgate/
```
on your command line, substituting in the appropriate file path.

### Hymod
Scripts for training the kernel ridge regression (KRR) surrogates for the Hymod model are located in `Hymod/train_krr/`. Run 
```
cd Hymod/train_krr/
bash train_10k.sh
bash train_100k.sh
```
to train the low- and high-quality surrogates with different hyperparameters. When all jobs have completed, run 
```
bash cross_val.sh
```
to evaluate the KRR models trained with different hyperparameters and save those with the best performance.

Scripts for running floodgate and the other sensitivity analysis methods we use for comparison on the Hymod model are located in `Hymod/analysis/`. Once both surrogates have been trained and cross-validated, run
```
cd ../analysis/
bash spf.sh
bash surrogate_10k.sh
bash surrogate_100k.sh
```
to submit batch jobs for applying each method to 1,000 independent datasets.

### CBMZ
Scripts for running floodgate and the other sensitivity analysis methods we use for comparison on the CBMZ model are located in `CBMZ/analysis/`. Run
```
cd $BASE_DIR/CBMZ/analysis/
bash run.sh
```

### Plotting results
Scripts for creating the plots in our paper are located in `plotting_scripts/`. Once all jobs from the previous two sections have completed, run
```
cd $BASE_DIR/plotting_scripts/
python3 Hymod.py
python3 CBMZ.py
```
to generate plots for both models. The figures will be saved in the `figs/` directory.
