# pyOPAC/active_learning/predict_with_uncertainty.py

import torch
import numpy as np
from torch.utils.data import DataLoader

def predict_with_uncertainty(model, dataset, num_samples=10):
    """
    Dummy function to return predictions and uncertainties.
    In a real scenario, you might use Monte Carlo Dropout or an ensemble.
    Here we perform one forward pass and assign a dummy uncertainty based on the standard deviation
    of the outputs across each sample's features.
    """
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    predictions = []
    uncertainties = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs = batch['descriptors']
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
            # Dummy uncertainty: standard deviation across the output dimensions for each sample
            uncertainties.append(np.std(outputs.cpu().numpy(), axis=1))
    predictions = np.concatenate(predictions, axis=0)
    uncertainties = np.concatenate(uncertainties, axis=0)
    return predictions, uncertainties
