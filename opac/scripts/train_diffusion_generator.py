"""
Script for training a diffusion model for molecular generation.
"""

import argparse
import pandas as pd
import torch
import os
import json
from opac.data.dataset import MoleculeDataset
from opac.models.diffusion import train_diffusion_generator, DiffusionMoleculeGenerator
from opac.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train a diffusion model for molecular generation.')
    parser.add_argument('--descriptors-file', type=str, required=True, help='CSV file containing descriptors.')
    parser.add_argument('--model-output', type=str, required=True, help='File to save the trained model.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    
    # Hyperparameters
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate for optimizer.')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Number of neurons in the hidden layer.')
    parser.add_argument('--num-timesteps', type=int, default=1000, help='Number of diffusion timesteps.')
    
    args = parser.parse_args()
    
    # Load descriptors
    df_descriptors = pd.read_csv(args.descriptors_file)
    descriptor_columns = [col for col in df_descriptors.columns if col != 'mol_id']
    descriptors = df_descriptors[descriptor_columns].to_dict('records')
    
    # Create dataset (no targets needed for generation)
    dataset = MoleculeDataset(descriptors, targets=None)
    descriptor_dim = dataset.input_dim
    
    logger.info(f"Training diffusion generator on {len(dataset)} molecules with descriptor dimension {descriptor_dim}.")
    
    # Train model
    model = train_diffusion_generator(
        dataset=dataset,
        descriptor_dim=descriptor_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        num_timesteps=args.num_timesteps
    )
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.model_output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created directory {output_dir}.")
    
    # Save the trained model state_dict
    torch.save(model.state_dict(), args.model_output)
    logger.info(f"Saved trained model to {args.model_output}.")
    
    # Save model parameters
    model_params = {
        'descriptor_dim': descriptor_dim,
        'hidden_dim': args.hidden_dim,
        'num_timesteps': args.num_timesteps,
        'model_type': 'diffusion_generator'
    }
    params_output = args.model_output + '.params.json'
    with open(params_output, 'w') as f:
        json.dump(model_params, f)
    logger.info(f"Saved model parameters to {params_output}.")
    
    logger.info("Training complete!")

if __name__ == '__main__':
    main()
