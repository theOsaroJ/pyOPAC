import os
import shutil
from creating_the_xyz.modify import modify_xyz_file

# Define file paths
input_xyz = "train.xyz"
modified_xyz = "train_m.xyz"
dest_dir = os.path.join("dataset", "training_xyz_files")
dest_file = os.path.join(dest_dir, "train.xyz")

# Modify the training XYZ file
modify_xyz_file(input_xyz, modified_xyz)
print(f"Modified file created: {modified_xyz}")

# Ensure the destination directory exists and copy the file there
os.makedirs(dest_dir, exist_ok=True)
shutil.copy(modified_xyz, dest_file)
print(f"Modified training file copied to {dest_file}")
