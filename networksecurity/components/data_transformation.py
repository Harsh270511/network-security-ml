
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
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    @staticmethod
    def get_data_transformer_object() -> Pipeline:
        try:
            logging.info("Creating data transformation pipeline")

            imputer = KNNImputer(
                **DATA_TRANSFORMATION_IMPUTER_PARAMS
            )

            processor = Pipeline(
                steps=[("imputer", imputer)]
            )

            return processor

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def initiate_data_transformation(
        self,
    ) -> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation")

            if not self.data_validation_artifact.validation_status:
                raise ValueError(
                    "Data validation failed. Transformation aborted."
                )

            train_df = self.read_data(
                self.data_validation_artifact.valid_train_file_path
            )
            test_df = self.read_data(
                self.data_validation_artifact.valid_test_file_path
            )

            if TARGET_COLUMN not in train_df.columns:
                raise ValueError(
                    f"Target column '{TARGET_COLUMN}' missing in training data."
                )

            if TARGET_COLUMN not in test_df.columns:
                raise ValueError(
                    f"Target column '{TARGET_COLUMN}' missing in testing data."
                )

            # Separate features and targets.
            X_train = train_df.drop(columns=[TARGET_COLUMN])
            y_train = train_df[TARGET_COLUMN].replace(-1, 0)

            X_test = test_df.drop(columns=[TARGET_COLUMN])
            y_test = test_df[TARGET_COLUMN].replace(-1, 0)

            if list(X_train.columns) != list(X_test.columns):
                raise ValueError(
                    "Training and testing feature columns do not match."
                )

            # Ensure all feature values are numeric.
            X_train = X_train.apply(pd.to_numeric, errors="raise")
            X_test = X_test.apply(pd.to_numeric, errors="raise")

            y_train = pd.to_numeric(y_train, errors="raise")
            y_test = pd.to_numeric(y_test, errors="raise")

            # Fit on training data only.
            preprocessor = self.get_data_transformer_object()
            preprocessor.fit(X_train)

            transformed_X_train = preprocessor.transform(X_train)
            transformed_X_test = preprocessor.transform(X_test)

            # Combine transformed features with target labels.
            train_array = np.column_stack(
                (transformed_X_train, y_train.to_numpy())
            )
            test_array = np.column_stack(
                (transformed_X_test, y_test.to_numpy())
            )

            # Save transformed arrays.
            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                array=train_array,
            )

            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                array=test_array,
            )

            # Save the fitted preprocessor.
            save_object(
                self.data_transformation_config.transformed_object_file_path,
                preprocessor,
            )

            artifact = DataTransformationArtifact(
                transformed_object_file_path=(
                    self.data_transformation_config.transformed_object_file_path
                ),
                transformed_train_file_path=(
                    self.data_transformation_config.transformed_train_file_path
                ),
                transformed_test_file_path=(
                    self.data_transformation_config.transformed_test_file_path
                ),
            )

            logging.info(
                f"Data transformation artifact: {artifact}"
            )

            return artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
