# VT-MRF-SPF: Variable Target Markov Random Field Scalable Particle Filter

This repository contains the Python implementation of the **VT-MRF-SPF** algorithm, as proposed in the paper [*VT-MRF-SPF: Variable Target Markov Random Field Scalable Particle Filter*](https://arxiv.org/abs/2404.18857) by N. Ning. 

The VT-MRF-SPF algorithm is an extension of the **VT-MRF-PF** algorithm for scalable spatiotemporal filtering in high-dimensional data. The **VT-MRF-PF** algorithm was proposed by Zia Khan, Tucker Balch, and Frank Dellaert in [MCMC-based Particle Filtering for Tracking a Variable Number of Interacting Targets](https://ieeexplore.ieee.org/abstract/document/1512059), *IEEE Transactions on Pattern Analysis and Machine Intelligence*.

## Core Functions

The implementation includes the following core files and their corresponding functions:

### 1. `external_functions.py`
This file contains utility functions used throughout the implementation of the VT-MRF-PF and VT-MRF-SPF algorithms. It includes helper functions for:
- Data preprocessing
- Particle management
- State propagation and other common external operations

### 2. `VTPF.py`
This file implements the **VT-MRF-PF** algorithm, which is an online learning algorithm used for tracking a variable number of interacting targets over time.

### 3. `VTSPF.py`
This file contains the implementation of the **VT-MRF-SPF** algorithm, an extension of the VT-MRF-PF with enhanced scalability for high-dimensional spatiotemporal problems.

### 4. `Likelihood_PF.py`
This script calculates the log-likelihood for the **VT-MRF-PF** algorithm, used to assess the performance of particle filtering for target tracking.

### 5. `Likelihood_SPF.py`
This file calculates the log-likelihood for the **VT-MRF-SPF** algorithm. It is used to quantify the accuracy of the scalable particle filtering.

## Implementation Folder

The `Implement` folder contains scripts and data used to evaluate and visualize the performance of the algorithms:

### 1. `W.csv`
This CSV file contains data for interaction weights used in the particle filtering processes.

### 2. `Full_Gaussian_eq_50.py`
This file generates full Gaussian state data for 50 targets, which is used as an input to test the performance of the implemented algorithms.

### 3. `check_full_Gaussian_eq.py`
This script is used to check the consistency and accuracy of the Gaussian data generated in `Full_Gaussian_eq_50.py`.

### 4. `plot_Gaussian_fullmatrix.py`
This script generates plots to visualize the results of the Gaussian data after running through the particle filtering algorithms. It is useful for examining the spatial relationships between targets and their interactions.

### 5. `plot_tracking_Gaussian_eq_50.py`
This file generates tracking visualizations of the 50 targets using the results from the VT-MRF-SPF and VT-MRF-PF algorithms, allowing for visual analysis of tracking performance over time.

## Getting Started

### Prerequisites

Before running the code, ensure you have the following Python libraries installed:

```bash
pip install numpy pandas matplotlib scipy
