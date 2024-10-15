import os
import time
import numpy as np
import pandas as pd
from importlib import reload
from scipy.stats import multivariate_normal as mvn, Covariance, binom, poisson
import matplotlib.pyplot as plt

new_directory = "/Users/patricianing/Desktop/Result"
os.chdir(new_directory)

M = [50, 100, 150, 200, 250, 300]
M_array = np.array(M).reshape(-1, 1)

ll_PF_fullmatrix = f"ll_VTPF_Gaussian_fullmatrix.npz"
ll_VTPF_data = np.load(ll_PF_fullmatrix)
ll_VTPF=ll_VTPF_data['arr_0'] / M_array
ll_VTPF[ll_VTPF == 0] = np.nan

ll_SPF_fullmatrix = f"ll_VTSPF_Gaussian_fullmatrix.npz"
ll_VTSPF_data = np.load(ll_SPF_fullmatrix)
ll_VTSPF=ll_VTSPF_data['arr_0'] / M_array
ll_VTSPF[ll_VTSPF == 0] = np.nan

sll_PF_fullmatrix = f"sll_VTPF_Gaussian_fullmatrix.npz"
sll_VTPF_data = np.load(sll_PF_fullmatrix)
sll_VTPF=sll_VTPF_data['arr_0'] / M_array
sll_VTPF[sll_VTPF == 0] = np.nan

sll_SPF_fullmatrix = f"sll_VTSPF_Gaussian_fullmatrix.npz"
sll_VTSPF_data = np.load(sll_SPF_fullmatrix)
sll_VTSPF=sll_VTSPF_data['arr_0'] / M_array
sll_VTSPF[sll_VTSPF == 0] = np.nan

######################################################################################################
# Set up the plot
plt.figure(figsize=(10, 6))

# Plot ll_VTPF data points (in blue)
for i in range(ll_VTPF.shape[0]):
    valid_indices = ~np.isnan(ll_VTPF[i, :])
    if np.any(valid_indices):
        plt.scatter([M[i]]*np.sum(valid_indices), ll_VTPF[i, valid_indices], color='blue', label='VT-MRF-PF' if i == 0 else "")

# Plot ll_VTSPF data points (in orange)
for i in range(ll_VTSPF.shape[0]):
    valid_indices = ~np.isnan(ll_VTSPF[i, :])
    if np.any(valid_indices):
        plt.scatter([M[i]]*np.sum(valid_indices), ll_VTSPF[i, valid_indices], color='orange', label='VT-MRF-SPF' if i == 0 else "")

# Calculate and plot mean for each row
mean_VTPF = np.nanmean(ll_VTPF, axis=1)  # Mean ignoring NaNs for ll_VTPF
mean_VTSPF = np.nanmean(ll_VTSPF, axis=1)  # Mean for ll_VTSPF

# Plot the average values for ll_VTPF (blue dashed line)
plt.plot(M, mean_VTPF, 'b-', linewidth=6,label='Avg. VT-MRF-PF', markersize=14)

# Plot the average values for ll_VTSPF (orange dashed line)
plt.plot(M, mean_VTSPF, 'orange', linewidth=6,linestyle='--', label='Avg. VT-MRF-SPF', markersize=14)

# Set y-axis range
#plt.ylim([mean_VTSPF[3]-30, mean_VTSPF[0]+30])

plt.xlabel('Spatial dimension', fontsize=24)
plt.ylabel('Spatial-scaled log-likelihood', fontsize=24)
plt.title('PF log-likelihood', fontsize=34)
# Adding legend
plt.legend(fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# Show the plot
plt.grid(True)
#plt.show()

#Save plot
plt.savefig('Gaussian_full_ll.png')

# Close the plot to prevent display
plt.close()

######################################################################################################
# Set up the plot
plt.figure(figsize=(10, 6))

# Plot sll_VTPF data points (in blue)
for i in range(sll_VTPF.shape[0]):
    valid_indices = ~np.isnan(sll_VTPF[i, :])
    if np.any(valid_indices):
        plt.scatter([M[i]]*np.sum(valid_indices), sll_VTPF[i, valid_indices], color='blue', label='VT-MRF-PF' if i == 0 else "")

# Plot sll_VTSPF data points (in orange)
for i in range(sll_VTSPF.shape[0]):
    valid_indices = ~np.isnan(sll_VTSPF[i, :])
    if np.any(valid_indices):
        plt.scatter([M[i]]*np.sum(valid_indices), sll_VTSPF[i, valid_indices], color='orange', label='VT-MRF-SPF' if i == 0 else "")

# Calculate and plot mean for each row
smean_VTPF = np.nanmean(sll_VTPF, axis=1)  # Mean ignoring NaNs for sll_VTPF
smean_VTSPF = np.nanmean(sll_VTSPF, axis=1)  # Mean for sll_VTSPF

# Plot the average values for sll_VTPF (blue dashed line)
plt.plot(M, smean_VTPF, 'b-', linewidth=6,label='Avg. VT-MRF-PF', markersize=14)

# Plot the average values for sll_VTSPF (orange dashed line)
plt.plot(M, smean_VTSPF, 'orange', linewidth=6,linestyle='--', label='Avg. VT-MRF-SPF', markersize=14)

# Set y-axis range
#plt.ylim([smean_VTPF[0]-30, smean_VTPF[0]+30])

plt.xlabel('Spatial dimension', fontsize=24)
plt.ylabel('Spatial-scaled log-likelihood', fontsize=24)
plt.title('SPF log-likelihood', fontsize=34)
# Adding legend
plt.legend(fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# Show the plot
plt.grid(True)
#plt.show()

#Save plot
plt.savefig('Gaussian_full_sll.png')

# Close the plot to prevent display
plt.close()