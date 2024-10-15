###############################################################################
#full spatial matrix to start from
#p=0.05, q=0.95
    # - p: The probability of indicator stay at 0 (i.e. not entering)
    # - q: The probability of indicator stay at 1 (i.e. staying)

import os
import time
import numpy as np
import pandas as pd
from importlib import reload
from scipy.stats import multivariate_normal as mvn, norm, Covariance, binom, poisson
import matplotlib.pyplot as plt

new_directory1 = "/scratch/user/patning/VTSPF/Data2"
#new_directory1 = "/home/patning/VTSPF/Code"
os.chdir(new_directory1)

Sresult_string = f"Gaussian_simulation_50_fullmatrix.npz"
Sloaded_data = np.load(Sresult_string)
True_psi_state = Sloaded_data['data2'] #(100, 50)

SPF_Sresult_string = f"Gaussian_TVSPF_50_0_fullmatrix.npz"
SPF_Sloaded_data = np.load(SPF_Sresult_string)
SPF_psi_state = SPF_Sloaded_data['data1'] #(100, 800, 50)

PF_Sresult_string = f"Gaussian_TVPF_50_0_fullmatrix.npz"
PF_Sloaded_data = np.load(PF_Sresult_string)
PF_psi_state = PF_Sloaded_data['data1'] #(100, 800, 50)

mean_SPF_psi_state = np.mean(SPF_psi_state, axis=1) #(100, 50)
mean_PF_psi_state = np.mean(PF_psi_state, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(True_psi_state[:, 0], label="True", color='blue', lw=4)
plt.plot(mean_SPF_psi_state[:, 0], label="VT-MRF-SPF", color='red', linestyle='--', lw=4)
plt.plot(mean_PF_psi_state[:, 0], label="VT-MRF-PF", color='green', linestyle='--', lw=4)

plt.title("Spatial Dimension 50", fontsize=24)
plt.xlabel("Time step", fontsize=24)
plt.ylabel("Latent State Value", fontsize=24)
plt.legend(fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Show the plot
plt.tight_layout()
plt.savefig('plot_tracking_Gaussian_eq_50.png')
plt.show()