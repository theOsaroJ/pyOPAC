"""
pyOPAC: A Python package for molecular property prediction and active learning.

This package provides:
- SOAP (Smooth Overlap of Atomic Positions) descriptors for size-invariant molecular representation
- Property prediction models using neural networks
- Active learning workflows for efficient data collection
- Tools for molecular data preprocessing and analysis
"""

__version__ = "0.1.0"
__author__ = "Etinosa Osaro and Yamil Colon"
__email__ = "eosaro@nd.edu, ycolon@nd.edu"

# Import main modules (these will be available at package level)
from opac.data import (
    compute_descriptors,
    get_all_species,
    read_xyz_files,
    MoleculeDataset,
)

__all__ = [
    'compute_descriptors',
    'get_all_species',
    'read_xyz_files',
    'MoleculeDataset',
    '__version__',
    '__author__',
    '__email__',
]
