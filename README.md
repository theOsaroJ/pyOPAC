# pyOPAC: Optimized Prediction and Active learning in Chemistry

## Overview

**pyOPAC** is a Python-based framework designed for molecular property prediction with integrated active learning. The project provides a modular pipeline to:

- **Modify XYZ Files:** Process multi-molecule XYZ files to add unique molecule identifiers.
- **Compute Descriptors:** Calculate molecular descriptors using libraries such as RDKit, ASE, and Open Babel.
- **Train Models:** Build and train neural network models (standard feedforward or diffusion models) for predicting molecular properties with optional uncertainty quantification.
- **Active Learning:** Iteratively enhance model performance by querying the most uncertain samples.
- **Prediction:** Predict properties for new molecules using both standard and active-learning–enhanced models.

The project is organized into a package named `opac` (with several subpackages) and a separate module directory `creating_the_xyz`. Interactive notebooks are available to run each step of the workflow.

## Directory Structure
```plaintext
pyOPAC/
├── creating_the_xyz/           # Modules for modifying XYZ files
│   └── modify.py
├── opac/                       # Main package directory
│   ├── active_learning/        # Active learning modules
│   │   ├── __init__.py
│   │   ├── active_learning.py
│   │   ├── al_loop.py
│   │   ├── data_loader.py
│   │   ├── logger.py
│   │   ├── predict_new_data.py
│   │   ├── predict_properties.py
│   │   ├── predict_with_uncertainty.py
│   │   ├── trainer.py
│   │   └── uncertainty_sampling.py
│   ├── data/                   # Data-related modules
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── descriptors.py
│   │   └── loader.py
│   ├── models/                 # Model definition and training modules
│   │   ├── __init__.py
│   │   ├── generator.py
│   │   ├── predictor.py
│   │   └── trainer.py
│   ├── scripts/                # Script modules for various workflow steps
│   │   ├── compute_descriptors.py
│   │   ├── generate_molecules.py
│   │   ├── preprocess_data.py
│   │   ├── run_active_learning.py
│   │   ├── train_model.py
│   │   └── predict_properties.py
│   └── utils/                  # Utility modules
│       └── logger.py
├── setup.py                    # Setup script for packaging
├── README.md                   # This file
└── (other files such as notebooks, etc.)
```


## Installation

### Quick Installation

**Install as a Python package (Recommended):**

```bash
# Install in development mode (editable - allows code editing)
pip install -e .

# Or standard installation
pip install .
```

This will install pyOPAC as a proper Python package with all command-line tools and Python API available.

### Prerequisites

- **Python:** 3.7 or higher
- **Required Packages (automatically installed):**
  - `numpy>=1.19.0`
  - `pandas>=1.2.0`
  - `scikit-learn>=0.24.0`
  - `torch>=1.8.0`
  - `ase>=3.21.0`
  - `dscribe>=2.0.0` (for SOAP descriptors)
  - `scipy>=1.6.0`

### Optional Dependencies

Install additional functionality:

```bash
# Install with all optional dependencies
pip install -e .[all]

# Install with development dependencies
pip install -e .[dev]
```

**External Libraries (install separately if needed):**
- **RDKit**: `conda install -c conda-forge rdkit` (not required for SOAP descriptors)
- **Open Babel**: `conda install -c conda-forge openbabel` (not required for SOAP descriptors)

**Note:** For SOAP descriptors (current implementation), only `dscribe` and `ase` are required. RDKit and Open Babel are optional.

### Verify Installation

After installation, verify everything works:

```bash
# Test package import
python -c "import opac; print(f'pyOPAC version: {opac.__version__}')"

# Test command-line tools
pyopac-compute-descriptors --help
pyopac-train --help

# Run test script
python test_installation.py
```

## Usage

### Command-Line Tools

After installation, you can use pyOPAC from the command line:

```bash
# Modify XYZ files
pyopac-modify-xyz input.xyz output.xyz

# Preprocess data and compute descriptors
pyopac-preprocess --input-dir path/to/xyz --targets-file targets.csv --output-descriptors descriptors.csv

# Compute descriptors for new molecules
pyopac-compute-descriptors --input-dir path/to/xyz --output-descriptors new_descriptors.csv

# Train model (standard feedforward NN)
pyopac-train --descriptors-file descriptors.csv --targets-file targets.csv --model-output model.pth --epochs 100

# Train diffusion model (with uncertainty quantification)
pyopac-train --descriptors-file descriptors.csv --targets-file targets.csv --model-output diffusion_model.pth --model-type diffusion --epochs 100

# Train diffusion generator for molecular generation
pyopac-train-diffusion-generator --descriptors-file descriptors.csv --model-output generator.pth --epochs 100

# Make predictions
pyopac-predict --model-file model.pth --descriptors-file new_descriptors.csv --predictions-output predictions.csv
```

### Python API

You can also use pyOPAC as a Python package:

```python
import opac
from opac.data.descriptors import compute_descriptors, get_all_species
from opac.data.loader import read_xyz_files

# Read molecules
molecules = read_xyz_files("path/to/xyz/files")

# Get all species for fixed-size descriptors (size-invariant)
all_species = get_all_species(molecules)

# Compute SOAP descriptors for each molecule
descriptors = []
for atoms in molecules:
    desc = compute_descriptors(atoms, species=all_species)  # Fixed species list ensures size-invariance
    descriptors.append(desc)
```

### Quick Start Examples

See the complete examples:

1. **Example** (`examples/`):
   ```bash
   cd examples/
   ```
   This runs a complete pipeline with SOAP descriptors.

2. **Gas Phase and Solution Phase Energy Example** (`examples/gas_phase_solution_phase_energy_example/`):
   ```bash
   cd examples/gas_phase_solution_phase_energy_example
   jupyter notebook examples_usage.ipynb
   ```
   This notebook demonstrates multi-target regression for predicting gas phase energy, solution phase energy, and solvation free energy.

### Data Preparation

1. **Modify XYZ Files:**  
   Use the command-line tool or `creating_the_xyz/modify.py` to process raw XYZ files.  
   ```bash
   pyopac-modify-xyz train_example.xyz train.xyz
   ```

2. **Compute Descriptors:**  
   Use the command-line tool or Python API to compute SOAP descriptors.  
   ```bash
   pyopac-compute-descriptors --input-dir training_xyz_files/ --output-descriptors descriptors.csv
   ```

### Model Training

pyOPAC supports multiple model types for property prediction:

#### 1. Standard Feedforward Neural Network (Default)
- **Use case:** Fast, deterministic property predictions
- **Command-line:**
  ```bash
  pyopac-train --descriptors-file descriptors.csv --targets-file targets.csv --model-output model.pth --epochs 100 --model-type standard
  ```
  
- **Python API:**
  ```python
  from opac.models.trainer import train_model
  from opac.data.dataset import MoleculeDataset
  
  dataset = MoleculeDataset(descriptors, targets)
  model = train_model(dataset, input_dim=descriptor_dim, output_dim=target_dim, epochs=100)
  ```

#### 2. Diffusion Model for Probabilistic Prediction
- **Use case:** Uncertainty-aware predictions with quantified uncertainty
- **Advantages:** Provides both mean predictions and uncertainty estimates
- **Command-line:**
  ```bash
  pyopac-train --descriptors-file descriptors.csv --targets-file targets.csv --model-output diffusion_model.pth --epochs 100 --model-type diffusion --num-timesteps 1000
  ```
  
- **Python API:**
  ```python
  from opac.models.diffusion import train_diffusion_property_predictor
  from opac.data.dataset import MoleculeDataset
  
  dataset = MoleculeDataset(descriptors, targets)
  model = train_diffusion_property_predictor(
      dataset=dataset,
      input_dim=descriptor_dim,
      output_dim=target_dim,
      epochs=100,
      num_timesteps=1000
  )
  
  # Predict with uncertainty
  mean_pred, std_pred = model.predict_with_uncertainty(descriptors_tensor, num_samples=10)
  ```

#### 3. Diffusion Model for Molecular Generation
- **Use case:** Generate new molecular descriptors (alternative to VAE)
- **Command-line:**
  ```bash
  pyopac-train-diffusion-generator --descriptors-file descriptors.csv --model-output diffusion_generator.pth --epochs 100
  ```
  
- **Python API:**
  ```python
  from opac.models.diffusion import train_diffusion_generator, DiffusionMoleculeGenerator
  
  # Train generator
  model = train_diffusion_generator(
      dataset=dataset,
      descriptor_dim=descriptor_dim,
      epochs=100
  )
  
  # Generate new molecules
  generated_descriptors = model.sample(batch_size=10)
  ```
  
The trained model and its parameters are saved automatically.

### Active Learning (Optional)

- **Active Learning Loop:**  
  Run the active learning notebook (e.g., `opac/active_learning/active_learning.py` or a corresponding notebook) to iteratively train the model while querying the most uncertain samples.  
  The active-learning–trained model is saved as `opac/models/al_trained_model.pth`.

### Prediction

- **Standard Prediction:**  
  Use the command-line tool:
  ```bash
  pyopac-predict --model-file model.pth --descriptors-file new_descriptors.csv --predictions-output predictions.csv
  ```
  
  Or in Python:
  ```python
  from opac.scripts.predict_properties import main as predict_main
  # Or use the Python API directly
  from opac.models.trainer import PropertyPredictor
  import torch
  
  model = PropertyPredictor(input_dim, hidden_dim, output_dim)
  model.load_state_dict(torch.load('model.pth'))
  model.eval()
  # Make predictions...
  ```

- **Active Learning Prediction:**  
  ```bash
  pyopac-active-learning --descriptors-file descriptors.csv --targets-file targets.csv ...
  ```

### Key Features

- **SOAP Descriptors**: Size-invariant, rotationally equivariant molecular descriptors
- **Fixed-Size Descriptors**: Works for molecules of any size (critical for inference)
- **Command-Line Tools**: Easy-to-use CLI for all operations
- **Python API**: Flexible programmatic interface
- **Active Learning**: Built-in active learning workflows

## Contributing

Contributions are welcome! Please open issues or submit pull requests on GitHub. Ensure that your changes include tests and updated documentation where necessary.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or support, please contact [eosaro@nd.edu]
