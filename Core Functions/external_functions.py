import os
import time
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal as mvn, norm, Covariance, binom, poisson

###############################################################################
def simulate_markov_chain(initial_state, p, q, num_steps):
    current_state = initial_state
    states = [current_state]
    transition_matrix = np.array([[p, 1-p],   # Probability of transitioning from 0 to 0 and 0 to 1
                                 [1-q, q]])  # Probability of transitioning from 1 to 0 and 1 to 1
    for _ in range(num_steps - 1):
        # Generate a random number to determine the next state
        random_number = np.random.rand()
        # Determine the next state based on the transition matrix
        if random_number < transition_matrix[current_state][0]:
            current_state = 0
        else:
            current_state = 1
        states.append(current_state)
    return states

###############################################################################
# simulation of dataset with varying dimension
def Simulation(T, M, W, p, q, distribution_type, varphi_initial, vartheta, bar_vartheta, sigma_tilde):
    """
    Generate simulated data
    Parameters:
    - T: The time dimension
    - M: The spatial dimension
    - W: The adjancy matrix of spatial interaction
    - p: The probability of indicator stay at 0 (i.e. not entering)
    - q: The probability of indicator stay at 1 (i.e. staying)
    - distribution_type: 'Gaussian' or 'Poisson'
    - varphi_initial: initial value of varphi
    - vartheta, bar_vartheta, sigma_tilde: model parameters
    Returns:
    A list of Y, psi, and varphi, all in size T*M.
    """
    ############generate the indicator matrix#####
    indicator_matrix=np.zeros((T,M))
    for j in range(0,M):
        indicator_matrix[:,j]=simulate_markov_chain(1, p, q, T)
    indicator_matrix=indicator_matrix.astype(int)
    #########Inital set up############
    phi=np.zeros((T, M))
    varphi=np.zeros((T, M))
    index_set_last=np.ones(M).astype(int)
    for i in range(0,T):
        #########generate phi component############
        index_set=indicator_matrix[i,:]
        W=W[0:M,0:M]
        W2=W[index_set == 1][:, index_set == 1]
        row_sum_W=np.sum(W2, axis=1)
        Denom_W=vartheta*row_sum_W+1-vartheta
        Denom_W_diag=np.diag(Denom_W)
        B_W_off_diag=vartheta*W2
        B_W_matrix_temp=Denom_W_diag-B_W_off_diag
        B_W_matrix=B_W_matrix_temp/sigma_tilde[i]
        #next is to generate the Gaussian r.v.
        Cov_phi = Covariance.from_precision(B_W_matrix)
        phi[i,index_set == 1] = mvn.rvs(mean=0, cov=Cov_phi, size=1)
        phi[i,index_set != 1] = np.nan
    #########generate varphi component############
        for k in range(M):
            if index_set_last[k] == 1:
                varphi[i,k] = norm.rvs(loc=bar_vartheta*varphi[i-1,k],scale = 0.1)   
            else:
                varphi[i,k] = norm.rvs(loc=bar_vartheta*varphi_initial[k],scale = 0.1) 
        index_set_last=index_set
        varphi[i,index_set != 1] = np.nan
    #########generate Y components############
    psi=phi+varphi #dimension is T*M
    Y=np.zeros((T, M))    
    if distribution_type == 'Gaussian':
        # Gaussian (Normal) random variable
        for i in range(0,T):
            Y[i,:] = mvn.rvs(psi[i,:], 0.01)
    elif distribution_type == 'Poisson':
        # Poisson random variable
        psi_non_nan=np.nan_to_num(psi, nan=0.5)
        psi_exp=np.exp(psi_non_nan)
        for i in range(0,T):
            Y[i,:] = np.random.poisson(psi_exp[i,:])
            Y[i,indicator_matrix[i,:]!= 1] = np.nan
    else:
        raise ValueError("Invalid distribution type. Choose 'Gaussian' or 'Poisson'.")
    ##########end of the function#########
    return Y, psi, varphi



  