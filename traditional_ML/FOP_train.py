"""
Random Forest Multi-Label Classifier Training Script

This script trains a Random Forest classifier for multi-label classification
using Mordred molecular descriptors and One-Hot Encoded labels.

Authors: Annika Siebenhaar, Francesco Ferri
Date: 2025
"""

import argparse
import ast
import logging
import pickle
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


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


class RFModelTrainer:
    """
    A class to handle training of Random Forest models for multi-label classification.
    """

    def __init__(
        self,
        n_estimators: int = 1100,
        min_samples_leaf: int = 3,
        min_samples_split: int = 7,
        random_state: int = 42
    ):
        """
        Initialize the Random Forest trainer.

        Args:
            n_estimators: Number of trees in the forest
            min_samples_leaf: Minimum samples required at leaf node
            min_samples_split: Minimum samples required to split node
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self.label_names = None

    def load_data(
        self,
        ohe_train_path: str,
        ohe_test_path: str,
        x_train_path: str,
        x_test_path: str,
        ohe_separator: str = ';',
        x_separator: str = ','
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load training and test data from CSV files.

        Args:
            ohe_train_path: Path to One-Hot Encoded training labels
            ohe_test_path: Path to One-Hot Encoded test labels
            x_train_path: Path to training features (Mordred descriptors)
            x_test_path: Path to test features (Mordred descriptors)
            ohe_separator: Separator for OHE files
            x_separator: Separator for feature files

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Loading One-Hot Encoded labels...")
        ohe_train = pd.read_csv(ohe_train_path, sep=ohe_separator)
        ohe_test = pd.read_csv(ohe_test_path, sep=ohe_separator)

        # Extract label names by analyzing the relationship between Descriptors and OneHotEncoding
        # Strategy: Find samples with only ONE descriptor active - these give definitive position mappings
        # Example: Descriptors=['earthy'] with OHE=[0,0,...,1,...,0] means that position maps to 'earthy'

        if 'Descriptors' in ohe_train.columns:
            logger.info("Mapping descriptors to OHE positions...")

            # Get the length of OHE array
            sample_ohe = ast.literal_eval(ohe_train['OneHotEncoding'].iloc[0])
            n_labels = len(sample_ohe)

            # Create mapping: position -> descriptor name
            position_to_descriptor = {}

            # First pass: Find samples with exactly ONE descriptor (definitive mappings)
            for idx in range(len(ohe_train)):
                try:
                    desc_str = ohe_train['Descriptors'].iloc[idx]
                    ohe_str = ohe_train['OneHotEncoding'].iloc[idx]

                    descriptors = ast.literal_eval(desc_str)
                    ohe_array = ast.literal_eval(ohe_str)

                    # Find positions where OHE is 1
                    active_positions = [i for i, val in enumerate(ohe_array) if val == 1]

                    # If exactly ONE descriptor and ONE position, we have a definitive mapping
                    if len(descriptors) == 1 and len(active_positions) == 1:
                        pos = active_positions[0]
                        desc = descriptors[0]
                        position_to_descriptor[pos] = desc

                except Exception as e:
                    continue

            logger.info(f"Found {len(position_to_descriptor)} definitive position mappings from single-descriptor samples")

            # Second pass: For remaining unmapped positions, use frequency analysis
            # Count how often each descriptor appears at each position
            position_descriptor_counts = {}
            for idx in range(len(ohe_train)):
                try:
                    desc_str = ohe_train['Descriptors'].iloc[idx]
                    ohe_str = ohe_train['OneHotEncoding'].iloc[idx]

                    descriptors = ast.literal_eval(desc_str)
                    ohe_array = ast.literal_eval(ohe_str)

                    active_positions = [i for i, val in enumerate(ohe_array) if val == 1]

                    # For each active position not yet mapped
                    for pos in active_positions:
                        if pos not in position_to_descriptor:
                            if pos not in position_descriptor_counts:
                                position_descriptor_counts[pos] = {}
                            # Count each descriptor that appears with this position
                            for desc in descriptors:
                                if desc not in position_descriptor_counts[pos]:
                                    position_descriptor_counts[pos][desc] = 0
                                position_descriptor_counts[pos][desc] += 1
                except:
                    continue

            # Assign most frequent descriptor to each unmapped position
            for pos, desc_counts in position_descriptor_counts.items():
                if desc_counts:
                    most_common_desc = max(desc_counts.items(), key=lambda x: x[1])[0]
                    position_to_descriptor[pos] = most_common_desc

            logger.info(f"Total mapped positions: {len(position_to_descriptor)}")

            # Build ordered label names list
            if len(position_to_descriptor) > 0:
                self.label_names = [''] * n_labels
                for pos, desc in position_to_descriptor.items():
                    if pos < n_labels:
                        self.label_names[pos] = desc

                # Fill unmapped positions with generic names
                for i in range(len(self.label_names)):
                    if self.label_names[i] == '':
                        self.label_names[i] = f'Label_{i}'

                if '' in [self.label_names[i] for i in range(n_labels)]:
                    logger.warning(f"Some OHE positions could not be mapped to descriptors")
            else:
                # Fallback: generate generic names
                self.label_names = [f'Label_{i}' for i in range(n_labels)]
                logger.warning(f"Could not map descriptors to positions, using generic names")
        else:
            # Fallback: generate generic names based on OHE length
            sample_ohe = ast.literal_eval(ohe_train['OneHotEncoding'].iloc[0])
            n_labels = len(sample_ohe)
            self.label_names = [f'Label_{i}' for i in range(n_labels)]
            logger.warning(f"'Descriptors' column not found, using generic names")

        # Show sample of label names
        logger.info(f"Total labels: {len(self.label_names)}")
        logger.info(f"Sample label names: {[f'{i}:{name}' for i, name in enumerate(self.label_names[:10])]}")

        # Parse One-Hot Encoded labels from string representation
        y_train = np.array([
            np.array(ast.literal_eval(f), dtype=int)
            for f in ohe_train['OneHotEncoding']
        ])
        y_test = np.array([
            np.array(ast.literal_eval(f), dtype=int)
            for f in ohe_test['OneHotEncoding']
        ])

        logger.info(f"Loaded labels - Train shape: {y_train.shape}, Test shape: {y_test.shape}")

        # Load feature data (Mordred descriptors)
        logger.info("Loading Mordred descriptors...")
        X_train_df = pd.read_csv(x_train_path, sep=x_separator)
        X_test_df = pd.read_csv(x_test_path, sep=x_separator)

        # Remove first column (index or identifier) and store feature names
        self.feature_names = X_train_df.columns[1:].tolist()
        X_train = X_train_df.iloc[:, 1:].values
        X_test = X_test_df.iloc[:, 1:].values

        logger.info(f"Loaded features - Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        logger.info(f"Stored {len(self.feature_names)} feature names for reproducibility")

        return X_train, X_test, y_train, y_test

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the Random Forest classifier.

        Args:
            X_train: Training features
            y_train: Training labels (multi-label)
        """
        logger.info("Initializing Random Forest classifier...")
        self.model = RandomForestClassifier(
            random_state=self.random_state,
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            min_samples_split=self.min_samples_split,
            n_jobs=-1  # Use all available cores
        )

        logger.info("Training model...")
        logger.info(f"Parameters: n_estimators={self.n_estimators}, "
                   f"min_samples_leaf={self.min_samples_leaf}, "
                   f"min_samples_split={self.min_samples_split}")

        self.model.fit(X_train, y_train)
        logger.info("Training completed successfully!")

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on input data.

        Args:
            X: Input features

        Returns:
            Tuple of (predictions, prediction_probabilities)
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet!")

        predictions = self.model.predict(X)
        predict_proba = self.model.predict_proba(X)

        # Extract probability for positive class (class 1) for each label
        proba_positive = np.transpose(np.array(predict_proba)[:, :, 1])

        return predictions, proba_positive

    def save_model(self, output_dir: str, model_name: str = "rf_model.pkl") -> None:
        """
        Save the trained model to disk.

        Args:
            output_dir: Directory to save the model
            model_name: Name of the model file
        """
        if self.model is None:
            raise ValueError("No model to save! Train the model first.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        model_path = output_path / model_name

        # Save model with feature names and label names
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'label_names': self.label_names,
            'n_estimators': self.n_estimators,
            'min_samples_leaf': self.min_samples_leaf,
            'min_samples_split': self.min_samples_split,
            'random_state': self.random_state
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {model_path}")

        # Also save model parameters and feature names to text file
        params_path = output_path / "model_params.txt"
        with open(params_path, 'w') as f:
            f.write(f"n_estimators: {self.n_estimators}\n")
            f.write(f"min_samples_leaf: {self.min_samples_leaf}\n")
            f.write(f"min_samples_split: {self.min_samples_split}\n")
            f.write(f"random_state: {self.random_state}\n")
            f.write(f"n_features: {len(self.feature_names) if self.feature_names else 'N/A'}\n")

        logger.info(f"Model parameters saved to {params_path}")

        # Save feature names separately
        if self.feature_names:
            features_path = output_path / "feature_names.txt"
            with open(features_path, 'w') as f:
                for feature in self.feature_names:
                    f.write(f"{feature}\n")
            logger.info(f"Feature names saved to {features_path}")

        # Save label names separately
        if self.label_names:
            labels_path = output_path / "label_names.txt"
            with open(labels_path, 'w') as f:
                for i, label in enumerate(self.label_names):
                    f.write(f"{i},{label}\n")
            logger.info(f"Label names saved to {labels_path}")


    def save_predictions(
        self,
        predictions: np.ndarray,
        proba: np.ndarray,
        output_dir: str,
        prefix: str = "Mordred_shortend-correlation_95"
    ) -> None:
        """
        Save predictions and probabilities to CSV files.

        Args:
            predictions: Model predictions
            proba: Prediction probabilities
            output_dir: Directory to save predictions
            prefix: Prefix for output filenames
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save predictions
        pred_filename = (f"{prefix}_RF_{self.n_estimators}_"
                        f"{self.min_samples_leaf}_{self.min_samples_split}_predict.csv")
        pred_path = output_path / pred_filename
        pd.DataFrame(predictions).to_csv(pred_path, sep=';', index=False)
        logger.info(f"Predictions saved to {pred_path}")

        # Save probabilities
        proba_filename = (f"{prefix}_RF_{self.n_estimators}_"
                         f"{self.min_samples_leaf}_{self.min_samples_split}_predict_proba.csv")
        proba_path = output_path / proba_filename
        pd.DataFrame(proba).to_csv(proba_path, sep=';', index=False)
        logger.info(f"Probabilities saved to {proba_path}")


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(
        description='Train Random Forest model for multi-label classification'
    )
    parser.add_argument(
        '--ohe-train',
        type=str,
        default=str(DATA_DIR / 'OHE_train_dataset_All_Food_6_shortend_recat.csv'),
        help='Path to One-Hot Encoded training labels'
    )
    parser.add_argument(
        '--ohe-test',
        type=str,
        default=str(DATA_DIR / 'OHE_test_dataset_All_Food_6_shortend_recat.csv'),
        help='Path to One-Hot Encoded test labels'
    )
    parser.add_argument(
        '--x-train',
        type=str,
        default=str(DATA_DIR / 'Mordred_train_dataset_shortend-correlation_95_All_Food_6.csv'),
        help='Path to training features (Mordred descriptors)'
    )
    parser.add_argument(
        '--x-test',
        type=str,
        default=str(DATA_DIR / 'Mordred_test_dataset_shortend-correlation_95_All_Food_6.csv'),
        help='Path to test features (Mordred descriptors)'
    )
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=1100,
        help='Number of trees in the forest'
    )
    parser.add_argument(
        '--min-samples-leaf',
        type=int,
        default=3,
        help='Minimum samples required at leaf node'
    )
    parser.add_argument(
        '--min-samples-split',
        type=int,
        default=7,
        help='Minimum samples required to split node'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(SCRIPT_DIR / 'models'),
        help='Directory to save trained model and predictions'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Initialize trainer
    trainer = RFModelTrainer(
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        min_samples_split=args.min_samples_split,
        random_state=args.random_state
    )

    # Load data
    X_train, X_test, y_train, y_test = trainer.load_data(
        ohe_train_path=args.ohe_train,
        ohe_test_path=args.ohe_test,
        x_train_path=args.x_train,
        x_test_path=args.x_test
    )

    # Train model
    trainer.train(X_train, y_train)

    # Make predictions on test set
    predictions, proba = trainer.predict(X_test)

    # Save model
    trainer.save_model(args.output_dir)

    # Save predictions
    trainer.save_predictions(predictions, proba, args.output_dir)

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()