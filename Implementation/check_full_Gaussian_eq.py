import os
import time
import numpy as np
import pandas as pd
from importlib import reload
from scipy.stats import multivariate_normal as mvn, Covariance, binom, poisson

M = [50, 100, 150, 200, 250, 300]
M=np.array(M)
t=100 #time dimension
Rep=5 #Number of replicates of test
ll_VTPF=np.zeros((len(M), Rep), dtype=int)
sll_VTPF=np.zeros((len(M), Rep), dtype=int)
ll_VTSPF=np.zeros((len(M), Rep), dtype=int)
sll_VTSPF=np.zeros((len(M), Rep), dtype=int)

new_directory = "/scratch/user/patning/VTSPF/Data2"
os.chdir(new_directory)

for i in range(len(M)):
    for r in range(Rep):
        try:
            Sresult_string = f"Gaussian_TVSPF_{M[i]}_{r}_fullmatrix.npz"
            Sloaded_data = np.load(Sresult_string)
            ll_VTSPF[i,r] = Sloaded_data['data5']
            sll_VTSPF[i,r] = Sloaded_data['data3']
            result_string = f"Gaussian_TVPF_{M[i]}_{r}_fullmatrix.npz"
            loaded_data = np.load(result_string)
            ll_VTPF[i,r] = loaded_data['data3']
            sll_VTPF[i,r] = loaded_data['data5']
        except Exception as e:
            continue

np.savez(f"ll_VTPF_Gaussian_fullmatrix.npz",ll_VTPF)
np.savez(f"ll_VTSPF_Gaussian_fullmatrix.npz",ll_VTSPF)
np.savez(f"sll_VTPF_Gaussian_fullmatrix.npz",sll_VTPF)
np.savez(f"sll_VTSPF_Gaussian_fullmatrix.npz",sll_VTSPF)

