

# TLS-ALS Forest Tree Dataset Tools

This repository contains essential scripts used for processing, filtering, and classifying high-resolution Terrestrial Laser Scanning (TLS) point cloud data of individual trees collected from the Shivalik Range, Haryana, India. These scripts support tasks such as wood-leaf separation and tree volume estimation.

## Repository Contents

### 1. `tls_separate.py`
Performs wood-leaf classification on TLS `.las` files using the [TLSeparation]([https://github.com/ekalinicheva/TLSeparation](https://tlseparation.github.io/documentation/)) algorithm.

- **Input:** `.las` point cloud files of individual trees
- **Output:** Text files containing point coordinates and wood-leaf classification labels
- **Usage:** Update the `input_directory` and `output_directory` to match your local file paths

### 2. `features_all.py`
Script for extracting a comprehensive set of geometric features from the TLS point cloud, which are used for machine learning classification.

- **Note:** Feature definitions and logic for computation are included in this file.

### 3. `FEATURE_INDEX.txt`
Contains indices of selected features (from `features_all.py`) that were used for training the Random Forest model.

### 4. `RF.py`
Performs batch classification of point cloud data using a trained Random Forest model.

- **Inputs:**
  - Directory with `.txt` files containing feature vectors
  - Trained `.pkl` model file
  - `FEATURE_INDEX.txt` to identify selected features
- **Outputs:** Text files containing classified labels
- **Execution:** Modify the `input_folder`, `output_folder`, `model_filepath`, and `feature_indices_filepath` accordingly.

### 5. `comp_vol_TreeQSM.m`
A MATLAB script using the TreeQSM method to compute volume from classified TLS data.

- **Note:** Requires MATLAB and the TreeQSM toolbox.

## Dependencies

- Python 3.x
- `numpy`
- `scikit-learn`
- `laspy`
- `tlseparation` (install from source: https://github.com/ekalinicheva/TLSeparation)
- MATLAB (for TreeQSM-based volume estimation)

## Applications

These scripts support the processing of TLS datasets for:

- Wood-leaf separation
- Tree volume estimation
- Feature extraction for machine learning
- Random Forest classification



