# pyOPAC/xyz_utils.py

import shutil

def modify_xyz(input_xyz, output_xyz):
    """
    Dummy function to modify an XYZ file.
    For demonstration, this simply copies the input file to the output.
    Replace with actual modification logic as needed.
    """
    shutil.copy(input_xyz, output_xyz)
    print(f"Modified {input_xyz} -> {output_xyz}")
