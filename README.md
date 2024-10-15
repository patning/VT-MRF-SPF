# VT-MRF-SPF: Variable Target Markov Random Field Scalable Particle Filter

This repository contains the Python implementation of the **VT-MRF-SPF** algorithm, as proposed in the paper [*VT-MRF-SPF: Variable Target Markov Random Field Scalable Particle Filter*](https://arxiv.org/abs/2404.18857) by Ning Ning. 

The VT-MRF-SPF algorithm is an extension of the **VT-MRF-PF** algorithm for scalable spatiotemporal filtering in high-dimensional data. The **VT-MRF-PF** algorithm was proposed by Zia Khan, Tucker Balch, and Frank Dellaert in [MCMC-based Particle Filtering for Tracking a Variable Number of Interacting Targets](https://ieeexplore.ieee.org/abstract/document/1512059), *IEEE Transactions on Pattern Analysis and Machine Intelligence*.

## Core Functions

The implementation includes the following core files and their corresponding functions:

### 1. `external_functions.py`
This file contains external functions includes the simulation function.

### 2. `VTPF.py`
This file implements the **VT-MRF-PF** algorithm, which is an online learning algorithm used for tracking a variable number of interacting targets over time.

### 3. `VTSPF.py`
This file contains the implementation of the **VT-MRF-SPF** algorithm, an extension of the VT-MRF-PF with enhanced scalability for high-dimensional spatiotemporal problems.

### 4. `Likelihood_PF.py`
This script calculates the log-likelihood that is used in the **VT-MRF-PF** algorithm.

### 5. `Likelihood_SPF.py`
This file calculates the log-likelihood that is used in the **VT-MRF-SPF** algorithm.

## Implementation Folder

The `Implement` folder contains scripts and data used to evaluate and visualize the performance of the algorithms:

### 1. `W.csv`
This CSV file contains real spatial data.

### 2. `Full_Gaussian_eq_50.py`
This file contains experiments for a 50-dimensional HSTMRF-VD model with equal target entering and staying probabilities under a complete spatial graph, assuming normally distributed observation errors.

### 3. `check_full_Gaussian_eq.py`
This script is used to check the experiment results.

### 4. `plot_Gaussian_fullmatrix.py`
This script generates plots of performance comparison of the VT-MRF-SPF and VT-MRF-PF algorithms across different spatial dimensions.

### 5. `plot_tracking_Gaussian_eq_50.py`
This file generates tracking visualizations of the 50 targets using the results from the VT-MRF-SPF and VT-MRF-PF algorithms.

## Getting Started

### Prerequisites

Before running the code, ensure you have the following Python libraries installed:

```bash
pip install numpy pandas matplotlib scipy
