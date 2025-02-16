#!/usr/bin/env python3

import argparse
import pandas as pd
import torch
import os
import json
from sklearn.model_selection import train_test_split

from pyOPAC.active_learning.data_loader import MoleculeDataset
from pyOPAC.models.trainer import train_model, evaluate_model
from pyOPAC.active_learning.logger import get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train a property prediction model.')
    parser.add_argument('--descriptors-file', type=str, required=True)
    parser.add_argument('--targets-file', type=str, required=True)
    parser.add_argument('--model-output', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--validation-size', type=float, default=0.2)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    args = parser.parse_args()

    df_descriptors = pd.read_csv(args.descriptors_file)
    df_targets = pd.read_csv(args.targets_file)
    df_descriptors['mol_id'] = df_descriptors['mol_id'].astype(int)
    df_targets['mol_id'] = df_targets['mol_id'].astype(int)
    df = pd.merge(df_descriptors, df_targets, on='mol_id')

    descriptor_columns = [col for col in df_descriptors.columns if col != 'mol_id']
    target_columns = [col for col in df_targets.columns if col != 'mol_id']

    train_df, test_df = train_test_split(df, test_size=args.validation_size, random_state=42)

    train_descriptors = train_df[descriptor_columns].to_dict('records')
    train_targets = train_df[target_columns].to_dict('records')
    train_dataset = MoleculeDataset(train_descriptors, train_targets)

    test_descriptors = test_df[descriptor_columns].to_dict('records')
    test_targets = test_df[target_columns].to_dict('records')
    test_dataset = MoleculeDataset(test_descriptors, test_targets)

    input_dim = train_dataset.input_dim
    output_dim = train_dataset.output_dim
    model = train_model(
        dataset=train_dataset,
        existing_model=None,
        input_dim=input_dim,
        output_dim=output_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        weight_decay=args.weight_decay
    )

    output_dir = os.path.dirname(args.model_output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created directory {output_dir}.")

    torch.save(model.state_dict(), args.model_output)
    logger.info(f"Saved trained model to {args.model_output}.")

    model_params = {
        'input_dim': input_dim,
        'hidden_dim': args.hidden_dim,
        'output_dim': output_dim
    }
    params_output = args.model_output + '.params.json'
    with open(params_output, 'w') as f:
        json.dump(model_params, f)
    logger.info(f"Saved model parameters to {params_output}.")

    test_loss, per_target_metrics = evaluate_model(model, test_dataset, batch_size=args.batch_size)
    logger.info(f"Test Loss (aggregated MSE): {test_loss:.4f}")
    for m in per_target_metrics:
        logger.info(f"[Target {m['target_index']}] MSE={m['MSE']:.4f}, MAE={m['MAE']:.4f}, R2={m['R2_Score']:.4f}")

if __name__ == '__main__':
    main()
