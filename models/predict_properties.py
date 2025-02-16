# pyOPAC/models/predict_properties.py

import argparse
import pandas as pd
import torch
import json
from opac3.models.trainer import PropertyPredictor
from opac3.active_learning.data_loader import MoleculeDataset
from opac3.active_learning.logger import get_logger

logger = get_logger(__name__)

def predict_properties(model_file, descriptors_file, predictions_output):
    """
    Load a trained model and predict properties for new molecules.
    """
    df_descriptors = pd.read_csv(descriptors_file)
    descriptor_columns = [col for col in df_descriptors.columns if col != 'mol_id']
    descriptors = df_descriptors[descriptor_columns].to_dict('records')
    dataset = MoleculeDataset(descriptors, targets=None)

    params_file = model_file + '.params.json'
    with open(params_file, 'r') as f:
        model_params = json.load(f)
    input_dim = model_params['input_dim']
    hidden_dim = model_params['hidden_dim']
    output_dim = model_params['output_dim']

    model = PropertyPredictor(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    predictions = []
    for sample in dataset:
        inputs = sample['descriptors'].unsqueeze(0)
        outputs = model(inputs)
        predictions.append(outputs.numpy().flatten())

    property_names = [f'Predicted_Property_{i+1}' for i in range(output_dim)]
    df_predictions = pd.DataFrame(predictions, columns=property_names)
    df_predictions['mol_id'] = df_descriptors['mol_id']
    df_predictions.to_csv(predictions_output, index=False)
    logger.info(f"Saved predictions to {predictions_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict properties using a trained model.")
    parser.add_argument('--model-file', type=str, required=True)
    parser.add_argument('--descriptors-file', type=str, required=True)
    parser.add_argument('--predictions-output', type=str, required=True)
    args = parser.parse_args()
    predict_properties(args.model_file, args.descriptors_file, args.predictions_output)
