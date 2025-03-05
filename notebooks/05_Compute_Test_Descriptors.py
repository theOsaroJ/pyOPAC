import os
import pandas as pd
from opac.data.loader import read_xyz_files
from opac.data.descriptors import compute_descriptors

# Directory containing test XYZ files
input_dir = os.path.join("dataset", "testing_xyz_files")

# Read molecules from the test XYZ files
molecules = read_xyz_files(input_dir)
print(f"Read {len(molecules)} molecules from {input_dir}")

# Compute descriptors for each test molecule
descriptors_list = []
for idx, atoms in enumerate(molecules):
    try:
        desc = compute_descriptors(atoms,  max_eigenvalues=22)
        desc['mol_id'] = idx
        descriptors_list.append(desc)
    except Exception as e:
        print(f"Failed to compute descriptors for molecule {idx}: {e}")

# Create a DataFrame and reorder columns so that 'mol_id' comes first
df_new = pd.DataFrame(descriptors_list)
cols = ['mol_id'] + [col for col in df_new.columns if col != 'mol_id']
df_new = df_new[cols]

# Save new descriptors to CSV
output_csv = os.path.join("dataset", "new_descriptors.csv")
df_new.to_csv(output_csv, index=False)
print(f"New descriptors saved to {output_csv}")
