import os
import pandas as pd
from opac.data.loader import read_xyz_files
from opac.data.descriptors import compute_descriptors

# Directory with training XYZ files
input_dir = os.path.join("dataset", "training_xyz_files")

# Read molecules from the XYZ files
molecules = read_xyz_files(input_dir)
print(f"Read {len(molecules)} molecules from {input_dir}")

# Compute descriptors for each molecule
descriptors_list = []
for idx, atoms in enumerate(molecules):
    try:
        desc = compute_descriptors(atoms, max_eigenvalues=22)
        desc['mol_id'] = idx  # assign molecule ID
        descriptors_list.append(desc)
    except Exception as e:
        print(f"Failed to compute descriptors for molecule {idx}: {e}")

# Create DataFrame and reorder columns so that 'mol_id' comes first
df_descriptors = pd.DataFrame(descriptors_list)
cols = ['mol_id'] + [col for col in df_descriptors.columns if col != 'mol_id']
df_descriptors = df_descriptors[cols]

# Save descriptors to CSV
output_csv = os.path.join("dataset", "descriptors.csv")
df_descriptors.to_csv(output_csv, index=False)
print(f"Descriptors saved to {output_csv}")
