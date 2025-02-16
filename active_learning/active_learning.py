# pyOPAC/active_learning/active_learning.py

import argparse
import pandas as pd
import torch
import os
from copy import deepcopy

from .al_loop import run_al_loop  # Example if you want to call something from al_loop
from .logger import get_logger
from .predict_with_uncertainty import predict_with_uncertainty
from .uncertainty_sampling import select_most_uncertain_samples
from pyOPAC.data.dataset import MoleculeDataset
from pyOPAC.models.trainer import train_model, evaluate_model

logger = get_logger(__name__)

def run_active_learning(descriptors_file, targets_file,
                        initial_train_size=10,
                        query_size=5,
                        iterations=10,
                        model_output='models/al_trained_model.pth',
                        hidden_dim=128,
                        epochs=100,
                        batch_size=32,
                        learning_rate=1e-3,
                        weight_decay=0.0):
    """
    Run the active learning loop. In each iteration, new uncertain samples are queried
    and added to the training set. The model is continuously trained (starting from
    previous weights). At the end, the final model and final training data are saved.
    """
    # Load data
    df_descriptors = pd.read_csv(descriptors_file)
    df_targets = pd.read_csv(targets_file)

    # Merge on 'mol_id'
    df = pd.merge(df_descriptors, df_targets, on='mol_id')
    descriptor_columns = [col for col in df_descriptors.columns if col != 'mol_id']
    target_columns = [col for col in df_targets.columns if col != 'mol_id']

    # Initialize labeled/unlabeled
    initial_train_df = df.sample(n=initial_train_size, random_state=42)
    unlabeled_df = df.drop(initial_train_df.index).reset_index(drop=True)

    max_iterations = max((len(df) - initial_train_size) // query_size, 0)
    iterations = min(iterations, max_iterations)
    if iterations == 0:
        logger.info("Not enough data for the specified iterations and query size.")
        return None

    model = None  # Start with no model

    for iteration in range(iterations):
        logger.info(f"Active Learning Iteration {iteration+1}/{iterations}")

        # Prepare train dataset
        train_descriptors = initial_train_df[descriptor_columns].to_dict('records')
        train_targets = initial_train_df[target_columns].to_dict('records')
        train_dataset = MoleculeDataset(train_descriptors, train_targets)

        # Prepare test dataset (everything not in initial_train_df)
        test_df = df.drop(initial_train_df.index).reset_index(drop=True)
        test_descriptors = test_df[descriptor_columns].to_dict('records')
        test_targets = test_df[target_columns].to_dict('records')
        test_dataset = MoleculeDataset(test_descriptors, test_targets)

        input_dim = len(descriptor_columns)
        output_dim = len(target_columns)

        # Train or continue training the model
        model = train_model(
            dataset=train_dataset,
            existing_model=model,
            input_dim=input_dim,
            output_dim=output_dim,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
            weight_decay=weight_decay
        )

        # Evaluate
        test_loss, per_target_metrics = evaluate_model(model, test_dataset, batch_size=batch_size)
        logger.info(f"Test Loss (aggregated MSE): {test_loss:.4f}")
        for m in per_target_metrics:
            logger.info(f"[Target {m['target_index']}] MSE={m['MSE']:.4f}, MAE={m['MAE']:.4f}, R2={m['R2_Score']:.4f}")

        # If unlabeled is empty, stop
        if unlabeled_df.empty:
            logger.info("No more unlabeled samples. Stopping AL.")
            break

        # Predict on unlabeled data
        unlabeled_descriptors = unlabeled_df[descriptor_columns].to_dict('records')
        unlabeled_dataset = MoleculeDataset(unlabeled_descriptors, targets=None)
        predictions, uncertainties = predict_with_uncertainty(model, unlabeled_dataset)

        current_query_size = min(query_size, len(unlabeled_df))
        query_indices = select_most_uncertain_samples(uncertainties, current_query_size)

        queried_samples = unlabeled_df.iloc[query_indices]
        initial_train_df = pd.concat([initial_train_df, queried_samples], ignore_index=True)
        unlabeled_df = unlabeled_df.drop(queried_samples.index).reset_index(drop=True)
        logger.info(f"Queried {len(queried_samples)} samples in iteration {iteration+1}.")

    # Save final model
    torch.save(model, model_output)
    logger.info(f"Active Learning completed. Final model saved to {model_output}")
    final_train_data_file = os.path.splitext(model_output)[0] + "_final_train.csv"
    initial_train_df.to_csv(final_train_data_file, index=False)
    logger.info(f"Saved final training data to {final_train_data_file}")

    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Active Learning for Molecular Property Prediction.")
    parser.add_argument('--descriptors-file', type=str, required=True)
    parser.add_argument('--targets-file', type=str, required=True)
    parser.add_argument('--initial-train-size', type=int, default=10)
    parser.add_argument('--query-size', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--model-output', type=str, default='models/al_trained_model.pth')
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    args = parser.parse_args()

    run_active_learning(
        descriptors_file=args.descriptors_file,
        targets_file=args.targets_file,
        initial_train_size=args.initial_train_size,
        query_size=args.query_size,
        iterations=args.iterations,
        model_output=args.model_output,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
