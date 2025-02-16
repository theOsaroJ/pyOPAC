# pyOPAC/active_learning/uncertainty_sampling.py

import numpy as np

def select_most_uncertain_samples(uncertainties, query_size):
    """
    Select indices of the samples with the highest uncertainty.
    
    Parameters:
      uncertainties (array-like): Uncertainty score for each sample.
      query_size (int): Number of samples to query.
    
    Returns:
      list: Indices of the selected samples.
    """
    uncertainties = np.array(uncertainties)
    sorted_indices = np.argsort(uncertainties)[::-1]  # descending order
    return sorted_indices[:query_size].tolist()
