###############################################################################
#full spatial matrix to start from
#p=0.05, q=0.95
    # - p: The probability of indicator stay at 0 (i.e. not entering)
    # - q: The probability of indicator stay at 1 (i.e. staying)

import os
import time
import numpy as np
import pandas as pd
import concurrent.futures
from importlib import reload
from scipy.stats import multivariate_normal as mvn, norm, Covariance, binom, poisson

#new_directory = "/Users/patricianing/Desktop/code"
new_directory = "/scratch/user/patning/VTSPF/Code2"
os.chdir(new_directory)

import external_functions
from external_functions import simulate_markov_chain, Simulation
import VTPF
from VTPF import TVPF
import VTSPF
from VTSPF import TVSPF
import Likelihood_SPF
from Likelihood_SPF import partition_sequence, SLL_SPF
import Likelihood_PF
from Likelihood_PF import LL_PF

reload(external_functions)
reload(VTPF)
reload(VTSPF)
reload(Likelihood_SPF)
reload(Likelihood_PF)

############setup the hyperparameters########
W = np.ones((500, 500), dtype=int)
np.fill_diagonal(W, 0)
m=50
t=100 #time dimension
NP=800 #Number of particles
Rep=5 #Number of replicates of test

#new_directory = "/Users/patricianing/Desktop/result"
new_directory = "/scratch/user/patning/VTSPF/Data2"
os.chdir(new_directory)

np.random.seed(1)
vartheta=np.random.rand(1)[0]
bar_vartheta=np.random.uniform(low=0.95, high=1.05, size=1) 
sigma_tilde=np.random.uniform(low=0.1, high=0.2, size=t) 
varphi_Gaussian=np.random.uniform(low=5, high=10, size=m)
varphi_Poisson=np.random.uniform(low=0.1, high=1, size=m)

Y_output, psi_output, varphi_output=Simulation(T=t, M=m, W=W, p=0.05, q=0.95, distribution_type = 'Gaussian', 
                                                varphi_initial=varphi_Gaussian, vartheta=vartheta, bar_vartheta=bar_vartheta, 
                                                sigma_tilde=sigma_tilde)
result_simulation = f"Gaussian_simulation_{m}_fullmatrix.npz"
np.savez(result_simulation, data1=Y_output, data2=psi_output, data3=varphi_output)

def task(r):
        #TVSPF
        psi_state_output, varphi_state_output, log_likelihood, elapsed_time=TVSPF(Y=Y_output, W=W, numberParticle=NP, 
        distribution_type='Gaussian', chunk_size=2, varphi_initial=varphi_Gaussian, vartheta=vartheta, bar_vartheta=bar_vartheta, sigma_tilde=sigma_tilde)
        LL=LL_PF(Y=Y_output, psi_state=psi_state_output, W=W, numberParticle=NP, distribution_type='Gaussian')
        result_TVSPF = f"Gaussian_TVSPF_{m}_{r}_fullmatrix.npz"
        np.savez(result_TVSPF, data1=psi_state_output, data2=varphi_state_output, data3=log_likelihood, data4=elapsed_time,
                data5=LL)
        #TVPF
        psi_state_output, varphi_state_output, log_likelihood, elapsed_time=TVPF(Y=Y_output, W=W, numberParticle=NP, 
                distribution_type='Gaussian', varphi_initial=varphi_Gaussian, vartheta=vartheta, bar_vartheta=bar_vartheta, sigma_tilde=sigma_tilde)
        SLL=SLL_SPF(Y=Y_output, psi_state=psi_state_output, W=W, numberParticle=NP, distribution_type='Gaussian', chunk_size=2)
        result_TVPF = f"Gaussian_TVPF_{m}_{r}_fullmatrix.npz"
        np.savez(result_TVPF, data1=psi_state_output, data2=varphi_state_output, data3=log_likelihood, data4=elapsed_time,
                data5=SLL)


if __name__ == "__main__":
    # Use ThreadPoolExecutor to run tasks concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=Rep+1) as executor:
        futures = {executor.submit(task, r): r for r in range(Rep)}  # Submit tasks
        for future in concurrent.futures.as_completed(futures):
            task_id = futures[future]
            try:
                print(f"Result from Task {task_id}")
            except Exception as exc:
                print(f'Task {task_id} generated an exception: {exc}')


