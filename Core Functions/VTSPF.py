import os
import time
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal as mvn, norm, Covariance, binom, poisson

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
# Variable Target Markov random field Scalable Particle Filter (VT-MRF-SPF)
###############################################################################
def TVSPF(Y, W, numberParticle, distribution_type, chunk_size, varphi_initial, vartheta, bar_vartheta, sigma_tilde):
    """
    Implementation of the VT-MRF-SPF algorithm
    Parameters:
    - Y: Spatiotemporal observations of varying dimension
    - W: The adjancy matrix of spatial interaction
    - numberParticle: Number of particles
    - distribution_type: 'Gaussian' or 'Poisson'
    - chunk_size: the block size
    - varphi_initial: initial value of varphi
    - vartheta, bar_vartheta, sigma_tilde: model parameters
    Returns:
    A list of psi_state, varphi_state, log_likelihood, elapsed_time
    """
    # initiate the run time checking
    start_time = time.time()
    # retrieve the indicator matrix
    indicator_matrix=np.where(np.isnan(Y), 0, 1)
    [T,M]=Y.shape
    index_set_last=np.ones(M).astype(int)
    # initialization of the state 
    varphi_last=np.tile(varphi_initial, (numberParticle, 1))
    # Set up variables
    varphi_state=np.zeros((T,numberParticle,M))
    psi_state=np.zeros((T,numberParticle,M))
    phi_state=np.zeros((numberParticle,M))
    log_likelihood=np.zeros(1)
    #general spatial interaction matrix
    W2=W[0:M,0:M]
    row_sum_W=np.sum(W2, axis=1)
    Denom_W=vartheta*row_sum_W+1-vartheta
    Denom_W_diag=np.diag(Denom_W)
    B_W_off_diag=vartheta*W2
    B_W_matrix_temp=Denom_W_diag-B_W_off_diag
    #next is to generate the covariance matrix
    Cov_phi = Covariance.from_precision(B_W_matrix_temp)
    # this is the main iteration of the scalable particle filter algorithm 
    for i in range(T):
        index_set=indicator_matrix[i,:]
        spatial_cov=Cov_phi.covariance*sigma_tilde[i]
        cov_matrix_ind=spatial_cov[index_set == 1][:, index_set == 1]
        mean_phi=np.zeros(np.sum(index_set))
        phiNew = mvn.rvs(mean=mean_phi, cov=cov_matrix_ind, size=numberParticle)
        #set up blocks
        block_index = partition_sequence(np.where(index_set == 1)[0], chunk_size)
        sublist_count = np.ceil(len(np.where(index_set == 1)[0])/chunk_size).astype(int)
        #define the weights
        newWeights=np.zeros((numberParticle,sublist_count))
        weightsStandardized=np.zeros((numberParticle,sublist_count))
        newWeights_npsum=np.zeros((sublist_count))
        #for log-likelihood 
        newWeights_l=np.zeros((numberParticle,sublist_count)) 
        newWeights_l_npsum=np.zeros((sublist_count))
        for j in range(numberParticle): 
            for k in range(M):
                if index_set_last[k] == 1:
                    varphi_state[i][j,k] = norm.rvs(loc=bar_vartheta*varphi_state[i-1][j,k],scale = 0.1)   
                else:
                    varphi_state[i][j,k] = norm.rvs(loc=bar_vartheta*varphi_initial[k],scale = 0.1)
            varphi_state[i][j,index_set != 1] = np.nan
            #each particle has its own phi and its own varphi 
            psi_state[i][j,index_set == 1] = phiNew[j,:]+varphi_state[i][j,index_set == 1]
            psi_state[i][j,index_set != 1] = np.nan
            phi_state[j,index_set == 1] = phiNew[j,:]
            # calculate the weights 
            if distribution_type == 'Gaussian':
                # Gaussian (Normal) random variable
                for k in range(sublist_count): 
                    newWeights_l[j,k]=mvn.pdf(Y[i,block_index[k]],mean=psi_state[i][j,block_index[k]],cov=0.01)
                    Cov_phi_B=spatial_cov[block_index[k]][:, block_index[k]]
                    newWeights[j,k]=newWeights_l[j,k]*mvn.pdf(phi_state[j,block_index[k]],mean=np.zeros(len(block_index[k])), cov=Cov_phi_B)       
            elif distribution_type == 'Poisson':
                # Poisson random variable
                for k in range(sublist_count): 
                    psi_exp=np.exp(psi_state[i][j,block_index[k]])
                    newWeights_l[j,k]=np.prod(poisson.pmf(Y[i,block_index[k]], psi_exp))
                    Cov_phi_B=spatial_cov[block_index[k]][:, block_index[k]]
                    newWeights[j,k]=newWeights_l[j,k]*mvn.pdf(phi_state[j,block_index[k]],mean=np.zeros(len(block_index[k])), cov=Cov_phi_B)       
            else:
                raise ValueError("Invalid distribution type. Choose 'Gaussian' or 'Poisson'.")
        #resample
        newWeights_npsum = np.sum(newWeights, axis=0)
        weightsStandardized=newWeights/newWeights_npsum
        for k in range(sublist_count):
            resampledStateIndex=np.random.choice(np.arange(numberParticle), numberParticle, p=weightsStandardized[:,k])
            varphi_state_temp=varphi_state[i][resampledStateIndex,:]
            varphi_state[i][:,block_index[k]]=varphi_state_temp[:,block_index[k]]    
        #update 
        index_set_last=index_set
        #calculate log-likelihood
        newWeights_l_npsum = np.sum(newWeights_l, axis=0)         
        newWeights_log=np.log(newWeights_l_npsum/numberParticle)
        log_likelihood += np.sum(newWeights_log)
    #calculate run time
    end_time = time.time()
    elapsed_time = end_time - start_time #run time in seconds
    ##########end of the function#########
    return psi_state, varphi_state, log_likelihood, elapsed_time

