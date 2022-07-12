import numpy as np
import pandas as pd

BASE_DIR = '~/Documents_offline/Research/floodgate/'

KRR_hyperparams = {
	100: (np.logspace(-8, 3, 30), np.logspace(-5, 4, 20)),
	1000: (np.logspace(-10, 2, 20), np.logspace(-5, 4, 20)),
	10000: (np.logspace(-13, -10, 4), np.logspace(-4, -2, 3)),
	50000: (np.logspace(-12, -10, 3), np.logspace(-4, -2, 3)),
	100000: (np.logspace(-12, -10, 3), np.logspace(-4, -2, 3))
}

Hymod_inputs = {
	"labels": ['Sm', 'beta', 'alfa', 'Rs', 'Rf'],
	"min": np.array([0, 0, 0, 0, 0.1]),
	"max": np.array([400, 2, 1, 0.1, 1]),
	"FORCING_PATH": BASE_DIR + 'Hymod/data/inputs/LeafCatch.txt'
}

conc_ranges = np.array(pd.read_csv(BASE_DIR + 'data/ranges/conc_range.csv'))
CBMZ_inputs = {
	"labels": 
	[
	  "H2SO4",
	  "HNO3",
	  "HCl",
	  "NH3",
	  "NO",
	  "NO2",
	  "NO3",
	  "N2O5",
	  "HONO",
	  "HNO4",
	  "O3",
	  "O1D",
	  "O3P",
	  "OH",
	  "HO2",
	  "H2O2",
	  "CO",
	  "SO2",
	  "CH4",
	  "C2H6",
	  "CH3O2",
	  "ETHP",
	  "HCHO",
	  "CH3OH",
	  "ANOL",
	  "CH3OOH",
	  "ETHOOH",
	  "ALD2",
	  "HCOOH",
	  "RCOOH",
	  "C2O3",
	  "PAN",
	  "ARO1",
	  "ARO2",
	  "ALK1",
	  "OLE1",
	  "API1",
	  "API2",
	  "LIM1",
	  "LIM2",
	  "PAR",
	  "AONE",
	  "MGLY",
	  "ETH",
	  "OLET",
	  "OLEI",
	  "TOL",
	  "XYL",
	  "CRES",
	  "TO2",
	  "CRO",
	  "OPEN",
	  "ONIT",
	  "ROOH",
	  "RO2",
	  "ANO2",
	  "NAP",
	  "XO2",
	  "XPAR",
	  "ISOP",
	  "ISOPRD",
	  "ISOPP",
	  "ISOPN",
	  "ISOPO2",
	  "API",
	  "LIM",
	  "DMS",
	  "MSA",
	  "DMSO",
	  "DMSO2",
	  "CH3SO2H",
	  "CH3SCH2OO",
	  "CH3SO2",
	  "CH3SO3",
	  "CH3SO2OO",
	  "CH3SO2CH2OO",
	  "SULFHOX",
	  "Num",  # #/cc
	  "DpgN", # um
	  "Sigmag", # -
	  "JHyst",                        # flag
	  "Water",                        # kg/m3
	  "SO4",                          # umol/m3
	  "PNO3",                         # umol/m3
	  "Cl",                           # umol/m3
	  "NH4",                          # umol/m3
	  "PMSA",                         # umol/m3
	  "Aro1",                         # umol/m3
	  "Aro2",                         # umol/m3
	  "Alk1",                         # umol/m3
	  "Ole1",                         # umol/m3
	  "PApi1",                        # umol/m3
	  "PApi2",                        # umol/m3
	  "Lim1",                         # umol/m3
	  "Lim2",                         # umol/m3
	  "CO3",                          # umol/m3
	  "Na",                           # umol/m3
	  "Ca",                           # umol/m3
	  "Oin",                          # ug/m3
	  "OC",                           # ug/m3
	  "BC",                           # ug/m3
	],
	"met_labels": ["T", "P", "RH", "COS_SZA"],
	"xmin": conc_ranges[0],
	"xmax": conc_ranges[1],
	"MODEL_PATH": "models/{}_FINAL_O3_multistep_16_20200118_NoNoiseNoSpinup_16_1_2b_256ls_16s_0d_1024_0.00128_o3pmfocusloss2_1RunNumber.h5"
}

