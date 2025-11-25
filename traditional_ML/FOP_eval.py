"""
Random Forest Model Prediction Script

This script reads SMILES strings from a file, computes Mordred molecular
descriptors, and predicts multi-label classifications using a trained
Random Forest model.

Authors: Annika Siebenhaar, Francesco Ferri
Date: 2025
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from mordred import Calculator, descriptors


# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get the absolute path to the script directory
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR / 'src'
DATA_DIR = SCRIPT_DIR / 'data'

# Add src directory to Python path
sys.path.insert(0, str(SRC_DIR))


class MordredFeatureComputer:
    """
    Computes Mordred molecular descriptors from SMILES strings.
    """

    def __init__(self, feature_names: List[str] = None):
        """
        Initialize the Mordred descriptor calculator.

        Args:
            feature_names: List of specific Mordred features to compute.
                          If None, all descriptors will be computed.
        """
        self.feature_names = feature_names
        self.calculator = Calculator(descriptors, ignore_3D=True)
        logger.info(f"Initialized Mordred calculator with {len(self.calculator.descriptors)} total descriptors")

    def smiles_to_mol(self, smiles: str) -> Chem.Mol:
        """
        Convert SMILES string to RDKit molecule object.

        Args:
            smiles: SMILES string

        Returns:
            RDKit molecule object or None if invalid
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES: {smiles}")
        return mol

    def compute_descriptors(self, smiles_list: List[str]) -> pd.DataFrame:
        """
        Compute Mordred descriptors for a list of SMILES.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            DataFrame with computed descriptors
        """
        logger.info(f"Computing Mordred descriptors for {len(smiles_list)} molecules...")

        # Convert SMILES to molecules
        mols = []
        valid_indices = []
        for idx, smiles in enumerate(smiles_list):
            mol = self.smiles_to_mol(smiles)
            if mol is not None:
                mols.append(mol)
                valid_indices.append(idx)
            else:
                mols.append(None)

        logger.info(f"Valid molecules: {len(valid_indices)}/{len(smiles_list)}")

        # Compute descriptors
        if len(valid_indices) == 0:
            raise ValueError("No valid SMILES strings found!")

        # Calculate all descriptors
        df_descriptors = self.calculator.pandas(mols)

        # Handle missing/invalid values
        # Replace NaN, inf, and problematic values with 0
        df_descriptors = df_descriptors.replace([np.inf, -np.inf], np.nan)
        df_descriptors = df_descriptors.fillna(0)

        # Convert to numeric, coercing errors
        for col in df_descriptors.columns:
            df_descriptors[col] = pd.to_numeric(df_descriptors[col], errors='coerce').fillna(0)

        logger.info(f"Computed {df_descriptors.shape[1]} descriptors")

        # Select only the required features if specified
        if self.feature_names is not None:
            logger.info(f"Selecting {len(self.feature_names)} required features...")

            # Check which features are available
            available_features = [f for f in self.feature_names if f in df_descriptors.columns]
            missing_features = [f for f in self.feature_names if f not in df_descriptors.columns]

            if missing_features:
                logger.warning(f"Missing {len(missing_features)} features from Mordred output")
                # Create missing columns with zeros
                for feature in missing_features:
                    df_descriptors[feature] = 0

            # Reorder columns to match training order
            df_descriptors = df_descriptors[self.feature_names]
            logger.info(f"Selected {len(self.feature_names)} features matching training data")

        return df_descriptors


class RFPredictor:
    """
    Loads a trained Random Forest model and makes predictions on new SMILES.
    """

    def __init__(self, model_path: str):
        """
        Initialize the predictor with a trained model.

        Args:
            model_path: Path to the saved model pickle file
        """
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        self.label_names = None
        self.model_params = {}
        self._load_model()

    def _load_model(self) -> None:
        """Load the trained model from disk."""
        logger.info(f"Loading model from {self.model_path}...")

        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)

        # Handle both old and new model formats
        if isinstance(model_data, dict):
            self.model = model_data['model']
            self.feature_names = model_data.get('feature_names', None)
            self.label_names = model_data.get('label_names', None)
            self.model_params = {
                'n_estimators': model_data.get('n_estimators'),
                'min_samples_leaf': model_data.get('min_samples_leaf'),
                'min_samples_split': model_data.get('min_samples_split'),
                'random_state': model_data.get('random_state')
            }
        else:
            # Old format - just the model
            self.model = model_data
            logger.warning("Loaded old model format without feature names")

        logger.info("Model loaded successfully!")
        if self.feature_names:
            logger.info(f"Model expects {len(self.feature_names)} features")
        if self.label_names:
            logger.info(f"Model has {len(self.label_names)} labels: {self.label_names}")

    def load_smiles(
        self,
        smiles_file: str,
        smiles_column: str = 'SMILES',
        separator: str = ','
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Load SMILES strings from a file.

        Args:
            smiles_file: Path to file containing SMILES
            smiles_column: Name of the column containing SMILES
            separator: File separator

        Returns:
            Tuple of (DataFrame with all data, list of SMILES strings)
        """
        logger.info(f"Loading SMILES from {smiles_file}...")

        # Try to read the file
        try:
            df = pd.read_csv(smiles_file, sep=separator)
        except:
            # Try with different separator
            try:
                df = pd.read_csv(smiles_file, sep=';')
            except:
                # Try tab-separated
                df = pd.read_csv(smiles_file, sep='\t')

        # Check if SMILES column exists
        if smiles_column not in df.columns:
            # Try to find a column with SMILES-like name
            smiles_cols = [col for col in df.columns if 'smiles' in col.lower()]
            if smiles_cols:
                smiles_column = smiles_cols[0]
                logger.info(f"Using column '{smiles_column}' for SMILES")
            else:
                # Assume first column is SMILES
                smiles_column = df.columns[0]
                logger.warning(f"SMILES column not found, using first column: '{smiles_column}'")

        smiles_list = df[smiles_column].tolist()
        logger.info(f"Loaded {len(smiles_list)} SMILES strings")

        return df, smiles_list

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on input features.

        Args:
            X: Input features (Mordred descriptors)

        Returns:
            Tuple of (binary predictions, prediction probabilities)
        """
        if self.model is None:
            raise ValueError("Model not loaded!")

        logger.info("Generating predictions...")

        # Get predictions
        predictions = self.model.predict(X)

        # Get probabilities
        predict_proba = self.model.predict_proba(X)

        # Extract probability for positive class (class 1) for each label
        proba_positive = np.transpose(np.array(predict_proba)[:, :, 1])

        logger.info("Predictions generated successfully!")

        return predictions, proba_positive

    def predict_from_smiles(
        self,
        smiles_file: str,
        output_file: str = None,
        smiles_column: str = 'SMILES',
        separator: str = ',',
        label_names: List[str] = None
    ) -> pd.DataFrame:
        """
        Complete pipeline: load SMILES, compute features, and predict.

        Args:
            smiles_file: Path to file containing SMILES
            output_file: Path to save predictions (optional)
            smiles_column: Name of column containing SMILES
            separator: File separator
            label_names: Names for the labels (optional)

        Returns:
            DataFrame with SMILES and predicted labels as text
        """
        # Load SMILES
        df_input, smiles_list = self.load_smiles(smiles_file, smiles_column, separator)

        # Compute Mordred descriptors
        feature_computer = MordredFeatureComputer(feature_names=self.feature_names)
        df_features = feature_computer.compute_descriptors(smiles_list)

        # Make predictions
        X = df_features.values
        predictions, probabilities = self.predict(X)

        # Create output dataframe
        df_output = df_input.copy()

        # Get number of labels
        n_labels = predictions.shape[1]

        # Use label names from model if available, otherwise use provided names or generic
        if label_names is None and self.label_names is not None:
            # Use label names stored in the model
            label_names = self.label_names
            logger.info(f"Using label names from model: {label_names}")
        elif label_names is None:
            # No label names available, use generic
            label_names = [f'Label_{i}' for i in range(n_labels)]
            logger.warning(f"No label names available, using generic names")
        elif len(label_names) != n_labels:
            # Provided label names don't match model
            logger.warning(f"Provided {len(label_names)} label names but model has {n_labels} labels.")
            if self.label_names is not None:
                logger.info(f"Using model's label names instead: {self.label_names}")
                label_names = self.label_names
            else:
                logger.warning(f"Using generic names")
                label_names = [f'Label_{i}' for i in range(n_labels)]

        # Create predicted labels column - only include activated labels
        predicted_labels_list = []
        for i in range(len(predictions)):
            active_labels = [label_names[j] for j in range(n_labels) if predictions[i, j] == 1]
            if active_labels:
                predicted_labels_list.append(', '.join(active_labels))
            else:
                predicted_labels_list.append('None')

        df_output['Predicted_Labels'] = predicted_labels_list

        # Optionally add individual label predictions and probabilities as separate columns
        # (commented out by default, uncomment if you want detailed breakdown)
        # for i, label_name in enumerate(label_names):
        #     df_output[f'{label_name}_prediction'] = predictions[:, i]
        #     df_output[f'{label_name}_probability'] = probabilities[:, i]

        # Save to file if specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df_output.to_csv(output_file, index=False)
            logger.info(f"Predictions saved to {output_file}")

        return df_output


def main():
    """Main prediction pipeline."""
    parser = argparse.ArgumentParser(
        description='Predict multi-label classifications from SMILES using trained RF model'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=str(SCRIPT_DIR / 'models' / 'rf_model.pkl'),
        help='Path to trained model pickle file'
    )
    parser.add_argument(
        '--smiles-file',
        type=str,
        required=True,
        help='Path to file containing SMILES strings'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default=str(SCRIPT_DIR / 'predictions' / 'predictions.csv'),
        help='Path to save predictions'
    )
    parser.add_argument(
        '--smiles-column',
        type=str,
        default='SMILES',
        help='Name of column containing SMILES'
    )
    parser.add_argument(
        '--separator',
        type=str,
        default=',',
        help='File separator (comma, semicolon, tab)'
    )
    parser.add_argument(
        '--label-names',
        type=str,
        nargs='+',
        default=None,
        help='Names for the predicted labels (space-separated)'
    )

    args = parser.parse_args()

    # Initialize predictor
    predictor = RFPredictor(model_path=args.model_path)

    # Ensure output directory exists
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Run prediction pipeline
    df_predictions = predictor.predict_from_smiles(
        smiles_file=args.smiles_file,
        output_file=str(output_path),
        smiles_column=args.smiles_column,
        separator=args.separator,
        label_names=args.label_names
    )

    # Display summary
    logger.info("\n" + "="*60)
    logger.info("Prediction Summary")
    logger.info("="*60)
    logger.info(f"Total molecules: {len(df_predictions)}")

    # Show sample predictions
    logger.info("\nSample predictions:")
    for idx in range(min(5, len(df_predictions))):
        smiles = df_predictions.iloc[idx][args.smiles_column]
        labels = df_predictions.iloc[idx]['Predicted_Labels']
        logger.info(f"  {smiles[:30]:<30} -> {labels}")

    # Count label frequencies
    if args.label_names:
        logger.info("\nLabel frequency:")
        all_predictions = df_predictions['Predicted_Labels'].str.split(', ')
        label_counts = {}
        for pred_list in all_predictions:
            if pred_list != ['None']:
                for label in pred_list:
                    label_counts[label] = label_counts.get(label, 0) + 1

        for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {label}: {count} molecules")

    logger.info("="*60)
    logger.info(f"\nResults saved to: {args.output_file}")
    logger.info("\nPrediction completed successfully!")


if __name__ == "__main__":
    main()