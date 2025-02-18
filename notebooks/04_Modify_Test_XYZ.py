import os
import shutil
from creating_the_xyz.modify import modify_xyz_file

# Define file paths for test XYZ
input_test_xyz = "test.xyz"
modified_test_xyz = "test_m.xyz"
dest_dir = os.path.join("dataset", "testing_xyz_files")
dest_file = os.path.join(dest_dir, "test.xyz")

# Modify the test XYZ file
modify_xyz_file(input_test_xyz, modified_test_xyz)
print(f"Modified test XYZ file created: {modified_test_xyz}")

# Copy the modified file to the testing directory
os.makedirs(dest_dir, exist_ok=True)
shutil.copy(modified_test_xyz, dest_file)
print(f"Modified test file copied to {dest_file}")
