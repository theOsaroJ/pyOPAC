import os
import pandas as pd
import torch
from copy import deepcopy

# Import required modules
from opac.data.dataset import MoleculeDataset
from opac.models.trainer import train_model, evaluate_model
from opac.active_learning.predict_with_uncertainty import predict_with_uncertainty
from opac.active_learning.uncertainty_sampling import select_most_uncertain_samples
from opac.utils.logger import get_logger

logger = get_logger(__name__)

# %%
# Set hyperparameters for active learning
initial_train_size = 1000
query_size = 5
requested_iterations = 2  # maximum number of active learning iterations
hidden_dim = 128
epochs = 50
batch_size = 32
learning_rate = 1e-3
weight_decay = 0.0

# File paths
descriptors_file = os.path.join("dataset", "descriptors.csv")
targets_file = os.path.join("dataset", "targets.csv")
model_output = os.path.join("saved_models", "al_trained_model.pth")
final_al_training_data = os.path.join("dataset", "final_al_training_data.csv")

# %%
# Load the descriptors and targets from CSV files
df_descriptors = pd.read_csv(descriptors_file)
df_targets = pd.read_csv(targets_file)

# Merge the data on 'mol_id'
df = pd.merge(df_descriptors, df_targets, on="mol_id")
descriptor_columns = [col for col in df_descriptors.columns if col != "mol_id"]
target_columns = [col for col in df_targets.columns if col != "mol_id"]

# %%
# Initialize the labeled (training) and unlabeled datasets
initial_train_df = df.sample(n=initial_train_size, random_state=42)
unlabeled_df = df.drop(initial_train_df.index).reset_index(drop=True)

logger.info(f"Initial training set size: {len(initial_train_df)}")
logger.info(f"Unlabeled set size: {len(unlabeled_df)}")

# %%
# Determine the maximum possible iterations based on available unlabeled data
max_possible_iterations = (len(df) - initial_train_size) // query_size
iterations = min(requested_iterations, max_possible_iterations)
if iterations == 0:
    logger.info("Not enough data for the specified iterations and query size.")
else:
    logger.info(f"Active learning will run for {iterations} iterations.")

# %%
# Active Learning Loop
model = None  # No pre-trained model to start with

for iteration in range(iterations):
    logger.info(f"--- Active Learning Iteration {iteration + 1}/{iterations} ---")
    
    # Create training dataset from the current labeled data
    train_descriptors = initial_train_df[descriptor_columns].to_dict("records")
    train_targets = initial_train_df[target_columns].to_dict("records")
    train_dataset = MoleculeDataset(train_descriptors, train_targets)
    
    # Create test dataset using the remaining data
    test_df = df.drop(initial_train_df.index).reset_index(drop=True)
    test_descriptors = test_df[descriptor_columns].to_dict("records")
    test_targets = test_df[target_columns].to_dict("records")
    test_dataset = MoleculeDataset(test_descriptors, test_targets)
    
    # Determine model dimensions
    input_dim = len(descriptor_columns)
    output_dim = len(target_columns)
    
    # Train or continue training the model using the current labeled dataset
    model = train_model(
        dataset=train_dataset,
        input_dim=input_dim,
        output_dim=output_dim,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_dim=hidden_dim,
        weight_decay=weight_decay
    )
    
    # Evaluate the model on the test set
    test_loss, per_target_metrics = evaluate_model(model, test_dataset, batch_size=batch_size)
    logger.info(f"Iteration {iteration + 1} - Test Loss: {test_loss:.4f}")
    for m in per_target_metrics:
        i = m["target_index"]
        logger.info(f"[Target {i}] MSE={m['MSE']:.4f}, MAE={m['MAE']:.4f}, R2={m['R2_Score']:.4f}")
    
    # If there is no more unlabeled data, stop the loop
    if unlabeled_df.empty:
        logger.info("No more unlabeled samples. Stopping active learning.")
        break
    
    # Use Monte Carlo Dropout to predict on unlabeled data and estimate uncertainty
    unlabeled_descriptors = unlabeled_df[descriptor_columns].to_dict("records")
    unlabeled_dataset = MoleculeDataset(unlabeled_descriptors, targets=None)
    predictions, uncertainties = predict_with_uncertainty(model, unlabeled_dataset, batch_size=batch_size)
    
    # Select the samples with the highest uncertainty
    current_query_size = min(query_size, len(unlabeled_df))
    query_indices = select_most_uncertain_samples(uncertainties, current_query_size)
    
    # Add the queried samples to the labeled dataset
    queried_samples = unlabeled_df.iloc[query_indices]
    initial_train_df = pd.concat([initial_train_df, queried_samples], ignore_index=True)
    
    # Remove the queried samples from the unlabeled dataset
    unlabeled_df = unlabeled_df.drop(queried_samples.index).reset_index(drop=True)
    logger.info(f"Iteration {iteration + 1}: Queried {len(queried_samples)} samples.")

# %%
# Save the final active learning model
output_dir = os.path.dirname(model_output)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)
    logger.info(f"Created directory {output_dir}.")
torch.save(model, model_output)
logger.info(f"Active Learning completed. Final model saved to {model_output}")

# Save the final active learning training dataset
initial_train_df.to_csv(final_al_training_data, index=False)
logger.info(f"Final AL training data saved to {final_al_training_data}")
