# Gas Phase and Solution Phase Energy Prediction Example

This example demonstrates how to use pyOPAC to predict gas phase energy, solution phase energy, and solvation free energy using SOAP descriptors.

## Prerequisites

1. **Install pyOPAC package** (from the repository root):
   ```bash
   cd /path/to/pyOPAC
   pip install -e .
   ```

2. **Prepare data**: This example uses data from `examples/1_gas_phase_solution_phase_energy/`

## Usage

### Run the Jupyter Notebook

Open `examples_usage.ipynb` in Jupyter and run all cells:

```bash
jupyter notebook examples_usage.ipynb
```

### What This Example Shows

- Reading molecular data from XYZ files
- Computing SOAP descriptors (size-invariant, rotationally equivariant)
- Loading target properties from DFT calculation results
- Training a multi-target regression model
- Making predictions and evaluating performance

## Data

- **Training molecules**: `examples/1_gas_phase_solution_phase_energy/train_example.xyz`
- **Target properties**: `results_train.dat` (gas phase energy, solution phase energy)
- **Computed properties**: Solvation free energy (calculated as difference)

## Notes

- This example works from any directory once pyOPAC is installed
- Paths are resolved relative to the notebook location
- The example automatically finds data files in the repository structure
