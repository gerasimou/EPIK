import pandas as pd
from tensorflow_probability import distributions as tfd
import tensorflow as tf
from scipy import stats
from scipy.spatial import distance
from math import log, log2, sqrt
import os
import numpy as np

from EPIK_Utils import get_files_using_substring, load_solutions_all

from EPIK_Utils import get_files_using_substring

data_dir = "data/oldFX"
problem_name = "FX_R1R2_p2p4p5_1000"
problem_dir = os.path.join(data_dir, problem_name)
pareto_dir = os.path.join(data_dir, problem_name, "ParetoFrontSet")
pareto_file_substring = "ParetoFrontSet"

ALGORITHMS = ["NSGAII", "SPEA2", "CMAES"]
NVARS      = 6

dfAlgos = []
for algo in ALGORITHMS:
    pf_files = get_files_using_substring(pareto_dir, pareto_file_substring + algo)

    dfAlgo = pd.DataFrame()
    for pf_file in pf_files:
        df_pf = load_solutions_all("", pf_file)

        pf_values = df_pf.iloc[:,NVARS:]

        dfAlgo = pd.concat([dfAlgo, pf_values], ignore_index=True)

    dfAlgos.append(dfAlgo)
    # print(dfAlgo.describe())

for i in range(len(ALGORITHMS)):
    print (ALGORITHMS[i], "\t", dfAlgos[i].mean(), dfAlgos[i].std())
