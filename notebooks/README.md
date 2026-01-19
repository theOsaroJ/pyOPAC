# pyOPAC Workflow Notebooks

This directory contains complete workflow notebooks and scripts for using pyOPAC. Each step has both a Jupyter notebook (`.ipynb`) and a Python script (`.py`) version.

## Complete Workflow

Follow these notebooks in order for a complete molecular property prediction pipeline:

### 1. Modify Training XYZ Files
- **Notebook**: `01_Modify_Training_XYZ.ipynb`
- **Script**: `01_Modify_Training_XYZ.py`
- **Purpose**: Process raw XYZ files to add unique molecule identifiers

### 2. Preprocess Training Data
- **Notebook**: `02_Preprocess_Training_Data.ipynb`
- **Script**: `02_Preprocess_Training_Data.py`
- **Purpose**: Compute SOAP descriptors for training molecules

### 3. Train Model
- **Notebook**: `03_Train_Model.ipynb`
- **Script**: `03_Train_Model.py`
- **Purpose**: Train property prediction model with default hyperparameters

### 3b. Train Model with Hyperparameter Tuning
- **Notebook**: `03_Train_Model_with_hyperparameters.ipynb`
- **Script**: `03_Train_Model_with_hyperparameters.py`
- **Purpose**: Perform grid search to find optimal hyperparameters

### 4. Modify Test XYZ Files
- **Notebook**: `04_Modify_Test_XYZ.ipynb`
- **Script**: `04_Modify_Test_XYZ.py`
- **Purpose**: Process test XYZ files to add molecule identifiers

### 5. Compute Test Descriptors
- **Notebook**: `05_Compute_Test_Descriptors.ipynb`
- **Script**: `05_Compute_Test_Descriptors.py`
- **Purpose**: Compute SOAP descriptors for test molecules (using same species list as training)

### 6. Predict Properties
- **Notebook**: `06_Predict_Properties.ipynb`
- **Script**: `06_Predict_Properties.py`
- **Purpose**: Make predictions using the trained model

### 7. Active Learning
- **Notebook**: `07_Active_Learning.ipynb`
- **Script**: `07_Active_Learning.py`
- **Purpose**: Run active learning loop to iteratively improve model

### 8. Predict with Active Learning Model
- **Notebook**: `08_Predict_with_al_model.ipynb`
- **Script**: `08_Predict_with_al_model.py`
- **Purpose**: Make predictions using the active-learning-enhanced model

## Usage

### Using Notebooks
1. Open Jupyter: `jupyter notebook`
2. Navigate to this directory
3. Run notebooks in order (01 → 02 → 03 → ...)

### Using Scripts
Run scripts from command line:
```bash
# Step 1: Modify XYZ files
python 01_Modify_Training_XYZ.py

# Step 2: Preprocess data
python 02_Preprocess_Training_Data.py

# Step 3: Train model
python 03_Train_Model.py

# ... and so on
```

## Prerequisites

1. **Install pyOPAC package**:
   ```bash
   pip install -e .
   ```

2. **Prepare your data**:
   - Training XYZ files
   - Test XYZ files (optional)
   - Target properties CSV file

## Data Structure

The notebooks expect the following directory structure:
```
your_workspace/
├── training_xyz_files/
│   └── train.xyz
├── testing_xyz_files/
│   └── test.xyz
├── saved_models/
├── dataset/
│   ├── descriptors.csv
│   └── targets.csv
└── notebooks/
    └── (these notebooks)
```

## Notes

- All scripts use relative paths and will create necessary directories
- SOAP descriptors are size-invariant and work for molecules of any size
- The species list from training data is saved and reused for test descriptors
- Models are saved in `saved_models/` directory
