import os
import time
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal as mvn, Covariance, binom, poisson

###############################################################################
###############################################################################
# Log-likelihood function with particle filter metric
###############################################################################
def LL_PF(Y, psi_state, W, numberParticle, distribution_type):
    """
    Generate log-likelihood
    Parameters:
    - Y: Spatiotemporal observations of varying dimension
    - psi_state: psi learned from one algorithm
    - W: The adjancy matrix of spatial interaction
    - numberParticle: Number of particles
    - distribution_type: 'Gaussian' or 'Poisson'
    Returns: log_likelihood
    """
    # retrieve the indicator matrix
    indicator_matrix=np.where(np.isnan(Y), 0, 1)
    [T,M]=Y.shape
    newWeights_l=np.zeros((T,numberParticle)) #for log-likelihood 
    # this is the main iteration of the particle filter algorithm 
    for i in range(T):
        index_set=indicator_matrix[i,:]
        for j in range(numberParticle): 
            if distribution_type == 'Gaussian':
                # Gaussian (Normal) random variable
                newWeights_l[i,j]=mvn.pdf(Y[i,index_set == 1],mean=psi_state[i][j,index_set == 1],cov=0.01)
            elif distribution_type == 'Poisson':
                # Poisson random variable
                psi_exp=np.exp(psi_state[i][j,index_set == 1])
                newWeights_l[i,j]=np.prod(poisson.pmf(Y[i,index_set == 1], psi_exp))
            else:
                raise ValueError("Invalid distribution type. Choose 'Gaussian' or 'Poisson'.")
    weights_row_avg = np.mean(newWeights_l, axis=1)
    log_likelihood = np.sum(np.log(weights_row_avg))
    ##########end of the function#########
    return log_likelihood

