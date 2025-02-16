# pyOPAC/models/trainer.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from opac3.active_learning.logger import get_logger
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = get_logger(__name__)

class PropertyPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PropertyPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.model(x)

def train_model(
    dataset,
    existing_model=None,
    input_dim=None,
    output_dim=None,
    epochs=100,
    batch_size=32,
    learning_rate=1e-3,
    hidden_dim=128,
    weight_decay=0.0
):
    """
    Train or continue training a multi-target (or single-target) regression model.
    If existing_model is provided, training continues from its weights.
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    if existing_model is None:
        model = PropertyPredictor(input_dim, hidden_dim, output_dim)
    else:
        model = existing_model

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    model.train()

    for epoch in range(epochs):
        total_loss_scalar = 0.0
        # Determine actual output dimension
        actual_output_dim = model.model[-1].out_features if hasattr(model.model[-1], 'out_features') else output_dim
        per_target_loss_accum = torch.zeros(actual_output_dim)
        num_samples = 0

        for batch in dataloader:
            inputs, targets = batch['descriptors'], batch['targets']
            optimizer.zero_grad()
            outputs = model(inputs)
            squared_errors = (outputs - targets) ** 2
            mse_batch = torch.mean(squared_errors, dim=0)
            final_loss = torch.mean(mse_batch)
            final_loss.backward()
            optimizer.step()

            bs = inputs.size(0)
            total_loss_scalar += final_loss.item() * bs
            per_target_loss_accum += mse_batch.detach() * bs
            num_samples += bs

        epoch_loss = total_loss_scalar / num_samples
        per_target_epoch_loss = per_target_loss_accum / num_samples

        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        loss_str = " ".join([f"T{i}={per_target_epoch_loss[i].item():.4f}" for i in range(actual_output_dim)])
        logger.info(f"     Per-Target: {loss_str}")

    return model

def evaluate_model(model, dataset, batch_size=32):
    """
    Evaluate the model on multi-target data.
    Returns:
      avg_loss: Aggregated MSELoss over the dataset.
      per_target_metrics: List of dictionaries with keys: 'target_index', 'MSE', 'MAE', 'R2_Score'.
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.MSELoss()
    model.eval()

    total_loss = 0.0
    all_targets = []
    all_predictions = []
    num_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch['descriptors'], batch['targets']
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            bs = inputs.size(0)
            total_loss += loss.item() * bs
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(outputs.cpu().numpy())
            num_samples += bs

    avg_loss = total_loss / num_samples
    all_targets = np.concatenate(all_targets, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    num_targets = all_targets.shape[1]
    per_target_metrics = []
    for i in range(num_targets):
        mse_i = mean_squared_error(all_targets[:, i], all_predictions[:, i])
        mae_i = mean_absolute_error(all_targets[:, i], all_predictions[:, i])
        r2_i = r2_score(all_targets[:, i], all_predictions[:, i])
        per_target_metrics.append({
            'target_index': i,
            'MSE': mse_i,
            'MAE': mae_i,
            'R2_Score': r2_i
        })

    return avg_loss, per_target_metrics
