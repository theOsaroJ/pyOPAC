import os
import json
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import itertools

# Import our unified dataset and training functions
from opac.data.dataset import MoleculeDataset
from opac.models.trainer import train_model, evaluate_model

# Load descriptors and targets
df_descriptors = pd.read_csv(os.path.join("dataset", "descriptors.csv"))
df_targets = pd.read_csv(os.path.join("dataset", "targets.csv"))

# Ensure 'mol_id' is numeric and merge on 'mol_id'
df_descriptors['mol_id'] = df_descriptors['mol_id'].astype(int)
df_targets['mol_id'] = df_targets['mol_id'].astype(int)
df = pd.merge(df_descriptors, df_targets, on='mol_id')

# Identify descriptor and target columns
descriptor_cols = [col for col in df_descriptors.columns if col != 'mol_id']
target_cols = [col for col in df_targets.columns if col != 'mol_id']

# Split data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_descriptors = train_df[descriptor_cols].to_dict('records')
train_targets = train_df[target_cols].to_dict('records')
test_descriptors = test_df[descriptor_cols].to_dict('records')
test_targets = test_df[target_cols].to_dict('records')

# Create datasets
train_dataset = MoleculeDataset(train_descriptors, train_targets)
test_dataset  = MoleculeDataset(test_descriptors, test_targets)

# Get dimensions for model input/output
input_dim = train_dataset.input_dim
output_dim = train_dataset.output_dim

# Example hyperparameter grid
param_grid = {
    'learning_rate': [0.001],
    'hidden_dim': [128],
    'weight_decay': [0.0, 1e-4]
}

# Hyperparameter tuning via grid search
best_loss = float('inf')
best_params = None
best_model_state = None

# Create combinations of hyperparameters
for lr, hidden, wd in itertools.product(param_grid['learning_rate'],
                                          param_grid['hidden_dim'],
                                          param_grid['weight_decay']):
    print(f"Training with lr={lr}, hidden_dim={hidden}, weight_decay={wd}")

    # Train the model using the current hyperparameters.
    # We pass 'existing_model=None' to train from scratch each time.
    model = train_model(
        dataset=train_dataset,
        input_dim=input_dim,
        output_dim=output_dim,
        epochs=100,              # You can adjust the number of epochs
        batch_size=64,
        learning_rate=lr,
        hidden_dim=hidden,
        weight_decay=wd
    )

    # Etestuate the model on the testidation set.
    test_loss, _ = evaluate_model(model, test_dataset, batch_size=32)
    print(f"Validation loss: {test_loss:.4f}")
    
    # If current configuration is better, store its parameters and model state.
    if test_loss < best_loss:
        best_loss = test_loss
        best_params = {'input_dim': input_dim, 'output_dim': output_dim,'learning_rate': lr, 'hidden_dim': hidden, 'weight_decay': wd}
        best_model_state = model.state_dict()  # Save the state_dict

print("Best Hyperparameters:")
print(best_params)
print(f"Best Validation Loss: {best_loss:.4f}")

torch.save(best_model_state, "saved_models/best_trained_model.pth")
with open("saved_models/best_trained_model.pth.params.json", "w") as f:
    json.dump(best_params, f)
