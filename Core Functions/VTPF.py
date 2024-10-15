import os
import time
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal as mvn, norm, Covariance, binom, poisson

###############################################################################
###############################################################################
# Variable Target Markov random field Particle Filter (VT-MRF-PF)
###############################################################################
def TVPF(Y, W, numberParticle, distribution_type, varphi_initial, vartheta, bar_vartheta, sigma_tilde):
    """
    Implementation of the VT-MRF-PF algorithm
    Parameters:
    - Y: Spatiotemporal observations of varying dimension
    - W: The adjancy matrix of spatial interaction
    - numberParticle: Number of particles
    - distribution_type: 'Gaussian' or 'Poisson'
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
    newWeights=np.zeros((T,numberParticle))
    newWeights_l=np.zeros((T,numberParticle)) #for log-likelihood
    weightsStandardized=np.zeros((T,numberParticle))
    #general spatial interaction matrix
    W2=W[0:M,0:M]
    row_sum_W=np.sum(W2, axis=1)
    Denom_W=vartheta*row_sum_W+1-vartheta
    Denom_W_diag=np.diag(Denom_W)
    B_W_off_diag=vartheta*W2
    B_W_matrix_temp=Denom_W_diag-B_W_off_diag
    #next is to generate the Covariance
    Cov_phi = Covariance.from_precision(B_W_matrix_temp)
    # this is the main iteration of the particle filter algorithm 
    for i in range(T): 
        index_set=indicator_matrix[i,:]
        spatial_cov=Cov_phi.covariance*sigma_tilde[i]
        cov_matrix_ind=spatial_cov[index_set == 1][:, index_set == 1]
        mean_phi=np.zeros(np.sum(index_set))
        phiNew = mvn.rvs(mean=mean_phi, cov=cov_matrix_ind, size=numberParticle)
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
        # calculate the weights 
            if distribution_type == 'Gaussian':
                # Gaussian (Normal) random variable
                newWeights_l[i,j]=mvn.pdf(Y[i,index_set == 1],mean=psi_state[i][j,index_set == 1],cov=0.01)
            elif distribution_type == 'Poisson':
                # Poisson random variable
                psi_exp=np.exp(psi_state[i][j,index_set == 1])
                newWeights_l[i,j]=np.prod(poisson.pmf(Y[i,index_set == 1], psi_exp))
            else:
                raise ValueError("Invalid distribution type. Choose 'Gaussian' or 'Poisson'.")
            newWeights[i,j]=newWeights_l[i,j]*mvn.pdf(phiNew[j,:],mean=np.zeros(np.sum(index_set)), cov=cov_matrix_ind)
        # standardize the weights  
        weightsStandardized[i,:]=newWeights[i,:]/(newWeights[i,:].sum())
        # resampling according to the probabilities stored in the weights
        resampledStateIndex=np.random.choice(np.arange(numberParticle), numberParticle, p=weightsStandardized[i,:])
        varphi_state[i]=varphi_state[i][resampledStateIndex,:]    
        #update
        index_set_last=index_set
    weights_row_avg = np.mean(newWeights_l, axis=1)
    log_likelihood = np.sum(np.log(weights_row_avg))
    #calculate run time
    end_time = time.time()
    elapsed_time = end_time - start_time #run time in seconds
    ##########end of the function#########
    return psi_state, varphi_state, log_likelihood, elapsed_time

