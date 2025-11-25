# Food odorant predictor Prediction with Random Forest

A machine learning model for multi-label classification of food odorants using Random Forest models and Mordred molecular descriptors.

**Authors:** Annika Siebenhaar, Francesco Ferri
**Year:** 2025

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Training a Model](#training-a-model)
  - [Making Predictions](#making-predictions)
- [Input Data Format](#input-data-format)
- [Output Format](#output-format)

## Overview

This project provides a complete pipeline for predicting odor profile from SMILES (Simplified Molecular Input Line Entry System) strings. The pipeline uses:

- **Mordred molecular descriptors** for feature representation
- **Random Forest classifier** for multi-label prediction
- **Scikit-learn** for machine learning
- **RDKit** for molecular processing

## Features

- Train Random Forest models on molecular descriptor data
- Predict multiple odor labels (multi-label classification)
- Automatic computation of Mordred descriptors from SMILES

## Requirements

- Python 3.12+
- RDKit
- scikit-learn
- pandas
- numpy
- mordredcommunity

## Installation

### Option 1: Using pip

1. **Clone the repository:**

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Create a `requirements.txt` file with the following contents:**
```
numpy==1.26.4
pandas==2.1.4
scikit-learn==1.7.2
rdkit==2023.9.1
mordredcommunity==2.0.6
```

4. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Option 2: Using conda

1. **Clone the repository:**

2. **Create conda environment:**
```bash
conda create -n FoodOdorantPredictor python=3.12
conda activate FoodOdorantPredictor
```

3. **Install RDKit via conda (recommended for RDKit):**
```bash
conda install -c conda-forge rdkit
```

4. **Install remaining packages:**
```bash
pip install pandas numpy scikit-learn mordredcommunity
```

## Usage

### Training a Model

To train a Random Forest model on your dataset:

**Basic usage:**
```bash
python FOP_train.py \
    --ohe-train data/OHE_train_dataset.csv \
    --ohe-test data/OHE_test_dataset.csv \
    --x-train data/Mordred_train_dataset.csv \
    --x-test data/Mordred_test_dataset.csv \
    --output-dir models/
```

**With custom parameters:**
```bash
python FOP_train.py \
    --ohe-train data/OHE_train_dataset.csv \
    --ohe-test data/OHE_test_dataset.csv \
    --x-train data/Mordred_train_dataset.csv \
    --x-test data/Mordred_test_dataset.csv \
    --n-estimators 1100 \
    --min-samples-leaf 3 \
    --min-samples-split 7 \
    --random-state 42 \
    --output-dir models/
```

### Making Predictions

To predict molecular properties from SMILES strings:

**Basic usage:**
```bash
python FOP_eval.py \
    --smiles-file input_smiles.csv \
    --output-file predictions/results.csv
```

**Prediction Parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model-path` | Path to trained model pickle file | `models/rf_model.pkl` |
| `--smiles-file` | Path to file containing SMILES strings | **Required** |
| `--output-file` | Path to save predictions | `predictions/predictions.csv` |
| `--smiles-column` | Name of column containing SMILES | `SMILES` |
| `--separator` | File separator (`,`, `;`, or `\t`) | `,` |
| `--label-names` | Names for predicted labels (space-separated) | Auto-detected from model |

## Input Data Format

CSV file with SMILES strings:
```csv
SMILES
CCO
c1ccccc1
CC(C)CO
```

Or with additional columns:
```csv
ID,SMILES,Name
1,CCO,Ethanol
2,c1ccccc1,Benzene
3,CC(C)CO,Isobutanol
```

## Output Format

The prediction script generates a CSV file with:
- All original columns from the input file
- `Predicted_Labels`: Comma-separated list of predicted labels

Example output:
```csv
SMILES,Predicted_Labels
CCO,"fruity, sweet"
c1ccccc1,"aromatic"
CC(C)CO,"fruity, alcoholic"
```
