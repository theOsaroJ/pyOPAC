# pyOPAC: Optimized Prediction and Active learning in Chemistry

## Overview

**pyOPAC** is a Python-based framework designed for molecular property prediction with integrated active learning. The project provides a modular pipeline to:

- **Modify XYZ Files:** Process multi-molecule XYZ files to add unique molecule identifiers.
- **Compute Descriptors:** Calculate molecular descriptors using libraries such as RDKit, ASE, and Open Babel.
- **Train Models:** Build and train neural network models for predicting molecular properties.
- **Active Learning:** Iteratively enhance model performance by querying the most uncertain samples.
- **Prediction:** Predict properties for new molecules using both standard and active-learning–enhanced models.

The project is organized into a package named `opac` (with several subpackages) and a separate module directory `creating_the_xyz`. Interactive notebooks are available to run each step of the workflow.

## Directory Structure
pyOPAC/ ├── creating_the_xyz/ # Modules for modifying XYZ files (ensure init.py if needed) │ └── modify.py ├── opac/ # Main package directory │ ├── active_learning/ # Active learning modules │ │ ├── init.py │ │ ├── active_learning.py │ │ ├── al_loop.py │ │ ├── data_loader.py │ │ ├── logger.py │ │ ├── predict_new_data.py │ │ ├── predict_properties.py │ │ ├── predict_with_uncertainty.py │ │ ├── trainer.py │ │ └── uncertainty_sampling.py │ ├── data/ # Data-related modules │ │ ├── init.py │ │ ├── dataset.py │ │ ├── descriptors.py │ │ └── loader.py │ ├── models/ # Model definition and training modules │ │ ├── init.py │ │ ├── generator.py │ │ ├── predictor.py │ │ └── trainer.py │ ├── scripts/ # Script modules for various workflow steps │ │ ├── compute_descriptors.py │ │ ├── generate_molecules.py │ │ ├── preprocess_data.py │ │ ├── run_active_learning.py │ │ ├── train_model.py │ │ └── predict_properties.py │ └── utils/ # Utility modules │ └── logger.py ├── setup.py # Setup script for packaging ├── README.md # This file └── (other files such as notebooks, etc.)


## Installation

### Prerequisites

- **Python:** 3.7 or higher.
- **Required Packages:**  
  - pandas  
  - torch  
  - scikit-learn  
  - ase  
- **Additional Libraries:**  
  RDKit and Open Babel (install these separately, as they are not available via PyPI).

## Usage

### Data Preparation

1. **Modify XYZ Files:**  
   Use the `creating_the_xyz/modify.py` module (or run the provided notebook) to process raw XYZ files (e.g., `train_example.xyz` and `test_example.xyz`).  
   The modified files will be stored in designated directories within your project.

2. **Compute Descriptors:**  
   Run the notebook for preprocessing (or use `opac/scripts/preprocess_data.py`) to generate descriptor CSV files from the XYZ files.

### Model Training

- **Train the Model:**  
  Execute the training notebook or run `opac/scripts/train_model.py` to merge descriptor and target data, split the data, and train your property prediction model.  
  The trained model and its parameters are saved in `opac/models`.

### Active Learning (Optional)

- **Active Learning Loop:**  
  Run the active learning notebook (e.g., `opac/active_learning/active_learning.py` or a corresponding notebook) to iteratively train the model while querying the most uncertain samples.  
  The active-learning–trained model is saved as `opac/models/al_trained_model.pth`.

### Prediction

- **Standard Prediction:**  
  Use the prediction notebook or run `opac/scripts/predict_properties.py` to predict molecular properties with the standard model.
- **Active Learning Prediction:**  
  Use the corresponding notebook (e.g., `opac/active_learning/predict_new_data.py` or a dedicated prediction notebook) to make predictions using the active-learning–trained model.

## Contributing

Contributions are welcome! Please open issues or submit pull requests on GitHub. Ensure that your changes include tests and updated documentation where necessary.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **RDKit:** [https://www.rdkit.org/](https://www.rdkit.org/)
- **ASE (Atomic Simulation Environment):** [https://wiki.fysik.dtu.dk/ase/](https://wiki.fysik.dtu.dk/ase/)
- **Open Babel:** [http://openbabel.org/](http://openbabel.org/)
- **PyTorch:** [https://pytorch.org/](https://pytorch.org/)

## Contact

For questions or support, please contact [eosaro@nd.edu]
