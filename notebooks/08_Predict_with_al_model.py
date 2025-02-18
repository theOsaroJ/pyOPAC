import os
import pandas as pd
import torch
from opac.data.dataset import MoleculeDataset  # Our unified dataset module
from opac.utils.logger import get_logger

logger = get_logger(__name__)

# Set the paths for the model, the input descriptors, and where to save predictions.

model_file = os.path.join("saved_models", "al_trained_model.pth")
descriptors_file = os.path.join("dataset", "new_descriptors.csv")
predictions_output = os.path.join("dataset", "new_predictions.csv")

# The CSV should contain a column `mol_id` plus descriptor columns.
df_descriptors = pd.read_csv(descriptors_file)
# Assume the descriptor columns are all columns except 'mol_id'
descriptor_columns = [col for col in df_descriptors.columns if col != "mol_id"]
descriptors = df_descriptors[descriptor_columns].to_dict("records")
print(f"Loaded descriptors for {len(descriptors)} molecules.")

# We create a dataset from the descriptors. Since we're only predicting, no targets are needed.
dataset = MoleculeDataset(descriptors, targets=None)

# The model was saved using `torch.save(model, model_file)`, so we load the entire model.
model = torch.load(model_file)
model.eval()
print(f"Loaded active learning model from {model_file}")

# For each sample in the dataset, we add a batch dimension to the descriptor tensor,
# run the model, and collect the predictions.
predictions = []
with torch.no_grad():
    for sample in dataset:
        # Add batch dimension to the descriptor tensor.
        inputs = sample["descriptors"].unsqueeze(0)
        outputs = model(inputs)
        predictions.append(outputs.numpy().flatten())

# Create a DataFrame with the predictions and the corresponding `mol_id`, then save to CSV.
if predictions:
    output_dim = predictions[0].shape[0]
    property_names = [f"Predicted_Property_{i+1}" for i in range(output_dim)]
else:
    property_names = []

df_preds = pd.DataFrame(predictions, columns=property_names)
df_preds["mol_id"] = df_descriptors["mol_id"]

df_preds.to_csv(predictions_output, index=False)
logger.info(f"Saved predictions to {predictions_output}")
print(f"Predictions saved to {predictions_output}")
