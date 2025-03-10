import argparse
import os
import pandas as pd
from opac.data.loader import read_xyz_files
from opac.data.descriptors import compute_descriptors
from opac.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Preprocess data and compute descriptors.')
    parser.add_argument('--input-dir', type=str, required=True, help='Directory containing XYZ files.')
    parser.add_argument('--targets-file', type=str, required=True, help='CSV file containing target properties.')
    parser.add_argument('--output-descriptors', type=str, required=True, help='Output file for descriptors (CSV).')
    args = parser.parse_args()

    # Read molecules
    molecules = read_xyz_files(args.input_dir)
    logger.info(f"Read {len(molecules)} molecules from {args.input_dir}.")

    # Compute descriptors
    descriptors_list = []
    for idx, atoms in enumerate(molecules):
        try:
            desc = compute_descriptors(atoms)
            desc['mol_id'] = idx  # Include mol_id
            descriptors_list.append(desc)
        except Exception as e:
            logger.warning(f"Failed to compute descriptors for molecule {idx}: {e}")

    # Convert to DataFrame and save
    df_descriptors = pd.DataFrame(descriptors_list)
    # Reorder columns to have 'mol_id' first
    columns = ['mol_id'] + [col for col in df_descriptors.columns if col != 'mol_id']
    df_descriptors = df_descriptors[columns]
    df_descriptors.to_csv(args.output_descriptors, index=False)
    logger.info(f"Saved descriptors to {args.output_descriptors}.")

if __name__ == '__main__':
    main()
