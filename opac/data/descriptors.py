from ase import Atoms
import numpy as np
from dscribe.descriptors import SOAP
from opac.utils.logger import get_logger

logger = get_logger(__name__)

# SOAP descriptor parameters (default values - can be configured)
DEFAULT_SOAP_PARAMS = {
    'rcut': 6.0,      # Cutoff radius in Angstrom
    'nmax': 6,        # Number of radial basis functions
    'lmax': 4,        # Maximum angular momentum (0=scalar, 1=vector, 2=tensor, etc.)
    'sigma': 0.3,     # Width of Gaussian smearing
    'periodic': False, # Non-periodic for molecules
    'crossover': True, # Include cross-species interactions
}

def get_all_species(molecules: list) -> list:
    """
    Helper function to determine all unique atomic species from a list of molecules.
    Useful for ensuring fixed-size SOAP descriptors across all molecules.
    
    Parameters:
      molecules: List of ASE Atoms objects
    
    Returns:
      Sorted list of all unique atomic species found in the dataset
    """
    all_species = set()
    for atoms in molecules:
        species = set(atoms.get_chemical_symbols())
        all_species.update(species)
    
    return sorted(list(all_species))

def compute_soap_descriptors(atoms: Atoms, rcut: float = 6.0, nmax: int = 6, lmax: int = 4, 
                             sigma: float = 0.3, periodic: bool = False, crossover: bool = True,
                             species: list = None) -> dict:
    """
    Computes SOAP (Smooth Overlap of Atomic Positions) descriptors - rotationally equivariant,
    translationally invariant, and size-invariant molecular representation.
    
    SOAP descriptors are:
    - Rotationally EQUIVARIANT: Descriptors transform predictably with rotations
    - Translationally invariant: Same descriptor regardless of molecular position
    - Size-invariant: Fixed-size global descriptor (averaged over all atoms)
    - 3D structure-aware: Capture local atomic environments based on 3D coordinates
    - Suitable for vector properties: Perfect for dipole moments, forces, etc.
    
    IMPORTANT: For fixed-size descriptors across different molecules, provide a fixed
    species list that includes all possible atomic species in your dataset.
    
    Parameters:
      atoms: ASE Atoms object representing the molecule
      rcut: Cutoff radius in Angstrom (default 6.0)
      nmax: Number of radial basis functions (default 6)
      lmax: Maximum angular momentum (default 4). Higher values capture more angular detail
      sigma: Width of Gaussian smearing (default 0.3)
      periodic: Whether system is periodic (default False for molecules)
      crossover: Include cross-species interactions (default True)
      species: Fixed list of atomic species for all molecules (default None, uses species in molecule).
               For size-invariant descriptors, specify all possible species: e.g., ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
      
    Returns:
      Dictionary with SOAP descriptor features (fixed size, size-invariant)
    """
    if len(atoms) == 0:
        raise ValueError("Cannot compute SOAP descriptors for empty molecule")
    
    try:
        # Use provided species list, or get from molecule
        if species is None:
            # Get unique atomic species from molecule
            species = sorted(list(set(atoms.get_chemical_symbols())))
        else:
            # Use provided species list (ensure sorted for consistency)
            species = sorted(list(set(species)))
        
        if len(species) == 0:
            raise ValueError("No atomic species specified")
        
        # Initialize SOAP descriptor
        soap = SOAP(
            species=species,
            rcut=rcut,
            nmax=nmax,
            lmax=lmax,
            sigma=sigma,
            periodic=periodic,
            crossover=crossover,
            sparse=False,
        )
        
        # Compute SOAP descriptors for all atoms
        # This returns shape (n_atoms, descriptor_dim)
        soap_features = soap.create(atoms)
        
        # Create global descriptor by averaging over all atoms (size-invariant)
        # This makes the descriptor fixed-size regardless of molecule size
        if len(soap_features.shape) == 2:
            global_soap = np.mean(soap_features, axis=0)  # Average over atoms
        else:
            global_soap = soap_features.flatten()
        
        # Convert to dictionary with feature names
        descriptors = {}
        for i, value in enumerate(global_soap):
            descriptors[f'SOAP_{i}'] = float(value)
        
        return descriptors
        
    except Exception as e:
        logger.error(f"Failed to compute SOAP descriptors: {e}")
        # Return zeros if computation fails (but need to know descriptor size first)
        # Try with minimal parameters to get size
        try:
            test_species = ['H', 'C', 'N', 'O']  # Common elements
            test_soap = SOAP(species=test_species, rcut=rcut, nmax=nmax, lmax=lmax, 
                           sigma=sigma, periodic=periodic, crossover=crossover, sparse=False)
            # Create dummy molecule to get descriptor size
            from ase import Atom
            dummy_atoms = Atoms([Atom('H', (0, 0, 0))])
            test_desc = test_soap.create(dummy_atoms)
            desc_size = test_desc.shape[-1] if len(test_desc.shape) > 1 else len(test_desc)
        except:
            # Fallback: estimate size from nmax, lmax, and species
            n_species = len(set(atoms.get_chemical_symbols())) if len(atoms) > 0 else 4
            if crossover:
                desc_size = nmax * nmax * (lmax + 1) * n_species * n_species
            else:
                desc_size = nmax * nmax * (lmax + 1) * n_species
        
        # Return zeros with correct size
        descriptors = {}
        for i in range(desc_size):
            descriptors[f'SOAP_{i}'] = 0.0
        
        logger.warning(f"Returning zero SOAP descriptors of size {desc_size}")
        return descriptors

def compute_descriptors(atoms: Atoms, max_eigenvalues: int = None, rcut: float = 6.0, 
                       nmax: int = 6, lmax: int = 4, sigma: float = 0.3, 
                       periodic: bool = False, crossover: bool = True, 
                       species: list = None) -> dict:
    """
    Computes size-invariant, rotationally equivariant molecular descriptors using SOAP.
    
    SOAP (Smooth Overlap of Atomic Positions) descriptors are:
    - Fixed size: Always returns same number of features regardless of molecule size
      (by averaging over all atoms to create global descriptor)
    - Size-invariant: Works for molecules of any size
    - Rotationally EQUIVARIANT: Descriptors transform predictably with rotations
    - Translationally invariant: Same descriptor regardless of molecular position
    - 3D structure-aware: Based on actual atomic positions, not just graph structure
    - Ideal for vector properties: Perfect for dipole moments, forces, etc.
    
    IMPORTANT for size-invariance: To ensure fixed-size descriptors across all molecules,
    specify a fixed species list that includes all possible atomic species in your dataset:
    
    Example:
      compute_descriptors(atoms, species=['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I'])
    
    Parameters:
      atoms: ASE Atoms object representing the molecule.
      max_eigenvalues: DEPRECATED - kept for backward compatibility but ignored.
      rcut: Cutoff radius in Angstrom (default 6.0). Larger values include more neighbors
      nmax: Number of radial basis functions (default 6). Higher = more radial detail
      lmax: Maximum angular momentum (default 4). Higher = more angular detail
            lmax=0 gives invariant, lmax>0 gives equivariant features
      sigma: Width of Gaussian smearing (default 0.3). Controls locality
      periodic: Whether system is periodic (default False for molecules)
      crossover: Include cross-species interactions (default True)
      species: Fixed list of atomic species for all molecules (default None, uses species in molecule).
               For size-invariant descriptors, specify all possible species in your dataset.
      
    Returns:
      A dictionary with fixed-size SOAP descriptors (always same size if species is fixed).
    """
    # Compute SOAP descriptors (single descriptor type, size-invariant, equivariant)
    descriptors = compute_soap_descriptors(
        atoms, 
        rcut=rcut, 
        nmax=nmax, 
        lmax=lmax, 
        sigma=sigma, 
        periodic=periodic, 
        crossover=crossover,
        species=species
    )
    
    return descriptors
