import os
import time
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal as mvn, Covariance, binom, poisson

###############################################################################
# implementation of sequence partition
def partition_sequence(sequence, chunk_size):
    """
    Partition a sequence into equal-length subsequences.
    Parameters:
    - sequence: The input sequence.
    - chunk_size: The size of each subsequence.
    Returns:
    A list of equal-length subsequences.
    """
    return [sequence[i:i+chunk_size] for i in range(0, len(sequence), chunk_size)]
###############################################################################

###############################################################################
# Log-likelihood function with scalable particle filter metric
###############################################################################
def SLL_SPF(Y, psi_state, W, numberParticle, distribution_type, chunk_size):
    """
    Generate log-likelihood
    Parameters:
    - Y: Spatiotemporal observations of varying dimension
    - psi_state: psi learned from one algorithm
    - W: The adjancy matrix of spatial interaction
    - numberParticle: Number of particles
    - distribution_type: 'Gaussian' or 'Poisson'
    - chunk_size: the block size
    Returns: log_likelihood
    """
    # retrieve the indicator matrix
    indicator_matrix=np.where(np.isnan(Y), 0, 1)
    [T,M]=Y.shape
    log_likelihood=np.zeros(1)
    # this is the main iteration of the particle filter algorithm 
    for i in range(T):
        index_set=indicator_matrix[i,:]
        #set up blocks
        block_index = partition_sequence(np.where(index_set == 1)[0], chunk_size)
        sublist_count = np.ceil(len(np.where(index_set == 1)[0])/chunk_size).astype(int)
        #for log-likelihood purpose
        newWeights_l=np.zeros((numberParticle,sublist_count)) 
        newWeights_l_npsum=np.zeros((sublist_count))
        for j in range(numberParticle): 
            # calculate the weights 
            if distribution_type == 'Gaussian':
                # Gaussian (Normal) random variable
                for k in range(sublist_count): 
                    newWeights_l[j,k]=mvn.pdf(Y[i,block_index[k]],mean=psi_state[i][j,block_index[k]],cov=0.01)
            elif distribution_type == 'Poisson':
                # Poisson random variable
                for k in range(sublist_count): 
                    psi_exp=np.exp(psi_state[i][j,block_index[k]])
                    newWeights_l[j,k]=np.prod(poisson.pmf(Y[i,block_index[k]], psi_exp))
            else:
                raise ValueError("Invalid distribution type. Choose 'Gaussian' or 'Poisson'.")
        #calculate log-likelihood
        newWeights_l_npsum = np.sum(newWeights_l, axis=0)         
        newWeights_log=np.log(newWeights_l_npsum/numberParticle)
        log_likelihood += np.sum(newWeights_log)
    ##########end of the function#########
    return log_likelihood

