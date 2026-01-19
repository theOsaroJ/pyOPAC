import os
import json
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

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

# Train the model
model = train_model(
    dataset=train_dataset,
    input_dim=input_dim,
    output_dim=output_dim,
    epochs=200,
    batch_size=64,
    learning_rate=0.001,
    hidden_dim=512,
    weight_decay=1e-4
)

# Save the trained model and parameters
model_path = os.path.join("saved_models", "trained_model.pth")
os.makedirs(os.path.dirname(model_path), exist_ok=True)
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

model_params = {"input_dim": input_dim, "hidden_dim": 512, "output_dim": output_dim}
params_path = model_path + ".params.json"
with open(params_path, "w") as f:
    json.dump(model_params, f)
print(f"Model parameters saved to {params_path}")

# Optionally, evaluate the model on the test set
test_loss, per_target_metrics = evaluate_model(model, test_dataset, batch_size=64)
print(f"Test Loss: {test_loss:.4f}")
for metric in per_target_metrics:
    print(metric)
