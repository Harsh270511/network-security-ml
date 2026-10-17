
import os
import sys

import pandas as pd
from scipy.stats import ks_2samp

from networksecurity.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.constant.training_pipeline import SCHEMA_FILE_PATH
from networksecurity.utils.main_utils.utils import (
    read_yaml_file,
    write_yaml_file,
)


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig,
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def validate_number_of_columns(
        self, dataframe: pd.DataFrame
    ) -> bool:
        try:
            schema_columns = self._schema_config.get("columns")

            # Support schemas represented as a list of column names
            # or a list of dictionaries containing column names.
            if isinstance(schema_columns, list):
                expected_columns = set()

                for column in schema_columns:
                    if isinstance(column, dict):
                        expected_columns.update(column.keys())
                    elif isinstance(column, str):
                        expected_columns.add(column)

                if expected_columns:
                    actual_columns = set(dataframe.columns)

                    if len(dataframe.columns) != len(expected_columns):
                        logging.error(
                            "Dataset column count does not match schema."
                        )
                        return False

                    if actual_columns != expected_columns:
                        logging.error(
                            "Dataset column names do not match schema."
                        )
                        return False

                    return True

            # Common alternative: column names are top-level schema keys.
            expected_columns = set(self._schema_config.keys())
            actual_columns = set(dataframe.columns)

            if actual_columns == expected_columns:
                return True

            logging.error(
                f"Expected columns: {expected_columns}; "
                f"actual columns: {actual_columns}"
            )
            return False

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def detect_dataset_drift(
        self,
        base_df: pd.DataFrame,
        current_df: pd.DataFrame,
        threshold: float = 0.05,
    ) -> bool:
        try:
            if list(base_df.columns) != list(current_df.columns):
                raise ValueError(
                    "Training and testing columns do not match."
                )

            report = {}
            drift_detected = False

            for column in base_df.columns:
                base_values = pd.to_numeric(
                    base_df[column], errors="raise"
                ).dropna()

                current_values = pd.to_numeric(
                    current_df[column], errors="raise"
                ).dropna()

                if base_values.empty or current_values.empty:
                    raise ValueError(
                        f"Column '{column}' has no usable numeric values."
                    )

                result = ks_2samp(base_values, current_values)

                column_drift = bool(result.pvalue < threshold)

                if column_drift:
                    drift_detected = True

                report[column] = {
                    "p_value": float(result.pvalue),
                    "drift_status": column_drift,
                }

            report_path = (
                self.data_validation_config.drift_report_file_path
            )

            report_directory = os.path.dirname(report_path)
            if report_directory:
                os.makedirs(report_directory, exist_ok=True)

            write_yaml_file(
                file_path=report_path,
                content=report,
            )

            logging.info(
                f"Dataset drift report saved: {report_path}"
            )

            # True means no statistically significant drift detected.
            return not drift_detected

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_path = (
                self.data_ingestion_artifact.trained_file_path
            )
            test_path = (
                self.data_ingestion_artifact.test_file_path
            )

            train_df = self.read_data(train_path)
            test_df = self.read_data(test_path)

            # Validate training and testing schemas.
            if not self.validate_number_of_columns(train_df):
                raise ValueError(
                    "Training dataset failed schema validation."
                )

            if not self.validate_number_of_columns(test_df):
                raise ValueError(
                    "Testing dataset failed schema validation."
                )

            if list(train_df.columns) != list(test_df.columns):
                raise ValueError(
                    "Training and testing column names or order differ."
                )

            logging.info("Schema validation completed successfully.")

            # Calculate drift and save the drift report.
            validation_status = self.detect_dataset_drift(
                base_df=train_df,
                current_df=test_df,
            )

            valid_train_path = (
                self.data_validation_config.valid_train_file_path
            )
            valid_test_path = (
                self.data_validation_config.valid_test_file_path
            )

            for file_path in (valid_train_path, valid_test_path):
                directory = os.path.dirname(file_path)
                if directory:
                    os.makedirs(directory, exist_ok=True)

            train_df.to_csv(valid_train_path, index=False)
            test_df.to_csv(valid_test_path, index=False)

            artifact = DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=valid_train_path,
                valid_test_file_path=valid_test_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=(
                    self.data_validation_config.drift_report_file_path
                ),
            )

            logging.info(
                f"Data validation completed: {artifact}"
            )

            return artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
