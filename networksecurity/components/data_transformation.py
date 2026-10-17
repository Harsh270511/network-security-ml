
import os
import sys

import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from networksecurity.constant.training_pipeline import (
    TARGET_COLUMN,
    DATA_TRANSFORMATION_IMPUTER_PARAMS,
)

from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact,
)

from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.utils.main_utils.utils import (
    save_numpy_array_data,
    save_object,
)


class DataTransformation:
    def __init__(
        self,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig,
    ):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """Read a validated CSV file."""
        try:
            return pd.read_csv(file_path)

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    @staticmethod
    def get_data_transformer_object() -> Pipeline:
        """Create the preprocessing pipeline."""
        logging.info(
            "Entered get_data_transformer_object method."
        )

        try:
            imputer = KNNImputer(
                **DATA_TRANSFORMATION_IMPUTER_PARAMS
            )

            logging.info(
                "Initialized KNNImputer with parameters: "
                f"{DATA_TRANSFORMATION_IMPUTER_PARAMS}"
            )

            processor = Pipeline(
                steps=[
                    ("imputer", imputer),
                ]
            )

            return processor

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def initiate_data_transformation(
        self,
    ) -> DataTransformationArtifact:
        logging.info(
            "Entered initiate_data_transformation method."
        )

        try:
            logging.info("Starting data transformation.")

            train_df = self.read_data(
                self.data_validation_artifact.valid_train_file_path
            )

            test_df = self.read_data(
                self.data_validation_artifact.valid_test_file_path
            )

            # Ensure that both datasets contain the target column.
            if TARGET_COLUMN not in train_df.columns:
                raise ValueError(
                    f"Target column '{TARGET_COLUMN}' is missing "
                    "from the training dataset."
                )

            if TARGET_COLUMN not in test_df.columns:
                raise ValueError(
                    f"Target column '{TARGET_COLUMN}' is missing "
                    "from the testing dataset."
                )

            # Separate training features and target.
            input_feature_train_df = train_df.drop(
                columns=[TARGET_COLUMN]
            )

            target_feature_train_df = train_df[TARGET_COLUMN].replace(
                -1, 0
            )

            # Separate testing features and target.
            # FIX: Do not pass axis=1 when using columns=.
            input_feature_test_df = test_df.drop(
                columns=[TARGET_COLUMN]
            )

            target_feature_test_df = test_df[TARGET_COLUMN].replace(
                -1, 0
            )

            # Ensure the feature columns match and use the same order.
            if set(input_feature_train_df.columns) != set(
                input_feature_test_df.columns
            ):
                raise ValueError(
                    "Training and testing datasets have different "
                    "feature columns."
                )

            input_feature_test_df = input_feature_test_df[
                input_feature_train_df.columns
            ]

            logging.info(
                f"Training features shape: "
                f"{input_feature_train_df.shape}"
            )
            logging.info(
                f"Testing features shape: "
                f"{input_feature_test_df.shape}"
            )

            # Fit preprocessing only on training features.
            preprocessor = self.get_data_transformer_object()

            transformed_input_train_feature = preprocessor.fit_transform(
                input_feature_train_df
            )

            transformed_input_test_feature = preprocessor.transform(
                input_feature_test_df
            )

            # Combine transformed features with their target labels.
            train_arr = np.c_[
                transformed_input_train_feature,
                target_feature_train_df.to_numpy(),
            ]

            test_arr = np.c_[
                transformed_input_test_feature,
                target_feature_test_df.to_numpy(),
            ]

            # Save transformed arrays.
            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                array=train_arr,
            )

            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                array=test_arr,
            )

            # Save the fitted preprocessor.
            save_object(
                self.data_transformation_config.transformed_object_file_path,
                preprocessor,
            )

            # Save a convenient copy for final inference.
            final_model_dir = "final_model"
            os.makedirs(final_model_dir, exist_ok=True)

            save_object(
                os.path.join(final_model_dir, "preprocessor.pkl"),
                preprocessor,
            )

            artifact = DataTransformationArtifact(
                transformed_object_file_path=(
                    self.data_transformation_config
                    .transformed_object_file_path
                ),
                transformed_train_file_path=(
                    self.data_transformation_config
                    .transformed_train_file_path
                ),
                transformed_test_file_path=(
                    self.data_transformation_config
                    .transformed_test_file_path
                ),
            )

            logging.info(
                f"Data transformation completed: {artifact}"
            )

            return artifact

        except Exception as e:
            logging.exception("Data transformation failed.")
            raise NetworkSecurityException(e, sys) from e
