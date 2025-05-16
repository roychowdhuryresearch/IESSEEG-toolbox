# EEG Data Preprocessing Pipeline

This directory contains scripts for preprocessing EEG data and metadata for various machine learning tasks. The preprocessing pipeline is organized into two main components: metadata preprocessing and EEG signal preprocessing.

## Directory Structure

```
preprocessing_release/
├── meta_data_preprocess/     # Metadata preprocessing scripts
├── eeg_preprocess/          # EEG signal preprocessing scripts
└── convert_meta.py          # Main metadata conversion script
```

## Metadata Preprocessing

The `meta_data_preprocess` directory contains scripts for handling metadata and creating train/test splits for different classification tasks:

### Key Scripts:
- `original_metas_to_tidy.py`: Converts original metadata files into a tidy format
- `create_case_vs_control_train_test_split.py`: Creates train/test splits for case vs. control classification
- `create_treatment_response_train_test_split.py`: Creates train/test splits for treatment response prediction
- `generate_test_meta.py`: Generates test metadata for evaluation

### Usage:
```bash
# Convert original metadata to tidy format
./meta_data_preprocess/original_metas_to_tidy.sh

# Create case vs. control splits
./meta_data_preprocess/create_case_vs_control_train_test_split.sh

# Create treatment response splits
./meta_data_preprocess/create_treatment_response_train_test_split.sh
```

## EEG Signal Preprocessing

The `eeg_preprocess` directory contains scripts for preprocessing EEG signals using different methods:

### Available Preprocessing Methods:
1. **Standard Preprocessing** (`preprocessing.py`)
2. **CBRAMOD Preprocessing** (`cbramod_preprocessing.py`)
3. **GNN-SSL Preprocessing** (`gnn_ssl_preprocessing.py`)
4. **BIOT Preprocessing** (`biot_preprocessing.py`)

### Usage:
```bash
# Run standard preprocessing
./eeg_preprocess/run_preprocessing.sh

# Run CBRAMOD preprocessing
./eeg_preprocess/run_cbramod_preprocessing.sh

# Run GNN-SSL preprocessing
./eeg_preprocess/run_gnn_ssl_preprocessing.sh

# Run BIOT preprocessing
./eeg_preprocess/run_biot_preprocessing.sh
```

## Main Metadata Conversion

The `convert_meta.py` script is the main script for converting and combining metadata from different sources. It:
- Merges multiple metadata files (MetaEEG.csv, MetaDataForMingjian.csv, long_eeg_meta.csv)
- Creates dictionaries for different recording types (awake/sleep)
- Handles case/control labels and treatment response labels
- Generates a final metadata file for use in baseline scripts

### Output Format:
The final metadata includes:
- Scenario information
- Filename mappings
- Patient IDs
- Labels (case/control, treatment response)
- Demographic information
- Recording type (awake/sleep)

## Requirements

- Python 3.x
- Required Python packages:
  - numpy
  - pandas
  - scipy
  - mne (for EEG processing)

## Notes

- All preprocessing scripts maintain the original sampling frequency (200Hz)
- The preprocessing pipeline supports both sleep and awake EEG recordings
- Different preprocessing methods are available for different model architectures
- The metadata preprocessing supports multiple classification tasks:
  - Case vs. Control
  - Treatment Response
  - Sleep vs. Awake
  - Combined classification tasks 