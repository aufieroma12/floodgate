# Floodgate for Sensitivity Analysis
This repository contains the source code accompanying our paper: Aufiero, Massimo and Janson, Lucas. "Surrogate-based global sensitivity analysis with statistical guarantees via floodgate." arXiv preprint [arXiv:2208.05885](https://arxiv.org/abs/2208.05885) (2022).

Any questions or concerns can be sent to Massimo Aufiero at aufieroma12@gmail.com.

## Instructions for Reproducing Results
The scripts in this repository were run on the FASRC Cannon cluster supported by the FAS Division of Science Research Computing Group at Harvard University. We have included `.sbatch` scripts and corresponding bash scripts for submitting parallel jobs that are compatible with any Linux-based cluster using the Slurm workload manager. While it is possible to run the Python scripts directly, serial computation time would be on the order of 10,000 hours.

### Setting up
After cloning this repository onto your local machine or compute cluster, change the `BASE_DIR` variable in `config/config.py` to the appropriate file path to this directory, and run 
```
export BASE_DIR=/path/to/floodgate/
```
on your command line, substituting in the appropriate file path.

For the simulations with the Hymod model, we use the implementation of the model provided in the SAFE Toolbox (Pianosi et al., 2015), which is available to request for download [here](https://www.safetoolbox.info/register-for-download/). Select Python as the language requested, and place the package in the `src/` directory once downloaded.

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

### CBM-Z
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

## References

[Pianosi, F., Sarrazin, F., Wagener, T. (2015), A Matlab toolbox for Global Sensitivity Analysis, Environmental Modelling & Software, 70, 80-85.](https://www.sciencedirect.com/science/article/pii/S1364815215001188)
- We used the Hymod implementation provided in the SAFE Toolbox in our simulations.

[Kelp, M. M., Jacob, D. J., Kutz, J. N., Marshall, J. D., & Tessum, C. W. (2020). Toward stable, general machine-learned models of the atmospheric chemical system. Journal of Geophysical Research: Atmospheres, 125, e2020JD032759.](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2020JD032759)
- Helper functions for the CBM-Z neural network surrogate contained in `src/build_nn.py` were borrowed from the [accompanying code](https://zenodo.org/record/4075312#.YvAC2C-B28V) for this paper. The first author of this paper, Makoto Kelp, also generously shared with us the data and pretrained models used in our experiments for the sake of transparency. 
