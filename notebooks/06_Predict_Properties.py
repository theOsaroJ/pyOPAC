import os
import json
import pandas as pd
import torch
from opac.data.dataset import MoleculeDataset
from opac.models.trainer import PropertyPredictor

# Load new descriptors
df_new = pd.read_csv(os.path.join("dataset", "new_descriptors.csv"))
descriptor_cols = [col for col in df_new.columns if col != 'mol_id']
descriptors = df_new[descriptor_cols].to_dict("records")
dataset = MoleculeDataset(descriptors, targets=None)

# Load model parameters
model_path = os.path.join("saved_models", "trained_model.pth")
params_path = model_path + ".params.json"
with open(params_path, "r") as f:
    model_params = json.load(f)
input_dim = model_params["input_dim"]
hidden_dim = model_params["hidden_dim"]
output_dim = model_params["output_dim"]

# Initialize and load the model
model = PropertyPredictor(input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load(model_path))
model.eval()

# Make predictions for each molecule
predictions = []
with torch.no_grad():
    for sample in dataset:
        # Add batch dimension to the tensor
        inputs = sample["descriptors"].unsqueeze(0)
        outputs = model(inputs)
        predictions.append(outputs.numpy().flatten())

# Create a DataFrame with predictions
property_names = [f"Predicted_Property_{i+1}" for i in range(output_dim)]
df_preds = pd.DataFrame(predictions, columns=property_names)
df_preds["mol_id"] = df_new["mol_id"]

# Save predictions to CSV
output_csv = os.path.join("dataset", "predictions.csv")
df_preds.to_csv(output_csv, index=False)
print(f"Predictions saved to {output_csv}")
