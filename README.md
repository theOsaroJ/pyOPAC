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
```plaintext
pyOPAC/
├── creating_the_xyz/
│   └── modify.py                # Module to modify XYZ files and add molecule IDs
├── data/
│   ├── targets.csv              # CSV file with target properties (user-provided)
│   ├── descriptors.csv          # Generated descriptors for training molecules
│   ├── new_descriptors.csv      # Generated descriptors for test molecules
│   ├── training_xyz_files/      # Directory for modified training XYZ files (train.xyz)
│   └── testing_xyz_files/       # Directory for modified test XYZ files (test.xyz)
├── models/
│   ├── trained_model.pth                # Saved trained property prediction model
│   ├── trained_model.pth.params.json    # Model parameters (input_dim, hidden_dim, output_dim)
│   ├── al_trained_model.pth             # Active-learning–trained model (optional)
│   ├── generator.py                     # (Optional) Model generator (e.g., VAE)
│   ├── predictor.py                     # Module for property prediction
│   └── trainer.py                       # Model training and evaluation functions
├── notebooks/
│   ├── 01_Modify_Training_XYZ.py        # Notebook to modify training XYZ files
│   ├── 02_Preprocess_Training_Data.py   # Notebook to compute training descriptors
│   ├── 03_Train_Model.py                # Notebook to train the property prediction model
│   ├── 04_Modify_Test_XYZ.py            # Notebook to modify test XYZ files
│   ├── 05_Compute_Test_Descriptors.py   # Notebook to compute test descriptors
│   ├── 06_Predict_Properties.py         # Notebook to predict properties using the trained model
│   ├── 07_Active_Learning.py            # Notebook to run the active learning loop
│   └── 08_Predict_with_AL_Model.py      # Notebook to predict properties with the AL model
├── dataset.py                # Unified dataset module (from opac3/data/dataset.py)
├── descriptors.py            # Module for computing molecular descriptors
├── loader.py                 # Module to read and process XYZ files
├── setup.py                  # Setup script for packaging the project
├── README.md                 # This file
└── utils/
    └── logger.py             # Logging configuration module
```


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

  Conda is preferred and for a simple installation process, run install.sh. Also activate conda environment each time you need to use pyOPAC

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
