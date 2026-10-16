import os
import sys

import pandas as pd
from scipy.stats import ks_2samp

from networksecurity.constant.training_pipeline import SCHEMA_FILE_PATH
from networksecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.utils.main_utils.utils import read_yaml_file, write_yaml_file


def get_schema_columns(schema: dict) -> list:
    """schema.yaml stores `columns` as a list of {name: dtype} entries -> return the names."""
    return [name for entry in schema["columns"] for name in entry.keys()]


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            self.expected_columns = get_schema_columns(self._schema_config)
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e: 
            raise NetworkSecurityException(e, sys) from e

    def validate_columns(self, dataframe: pd.DataFrame) -> bool:
        """True only if the dataframe has exactly the columns listed in schema.yaml."""
        missing = set(self.expected_columns) - set(dataframe.columns)
        extra = set(dataframe.columns) - set(self.expected_columns)
        if missing:
            logging.error(f"Missing columns: {sorted(missing)}")
        if extra:
            logging.error(f"Unexpected columns: {sorted(extra)}")
        return not missing and not extra

    def validate_numeric_columns(self, dataframe: pd.DataFrame) -> bool:
        """Every column listed under `numerical_columns` must be numeric."""
        bad = [c for c in self._schema_config.get("numerical_columns", [])
               if c in dataframe.columns and not pd.api.types.is_numeric_dtype(dataframe[c])]
        if bad:
            logging.error(f"Columns that should be numeric but are not: {bad}")
        return not bad

    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame) -> bool:
        """
        Kolmogorov-Smirnov test per column. Writes report.yaml and returns True if
        drift was found in ANY column. Drift is reported, but it does not stop the
        pipeline (with 30 columns, a few chance hits at p<0.05 are normal).
        """
        try:
            threshold = self.data_validation_config.drift_threshold
            report, drift_found = {}, False
            for column in base_df.columns:
                p_value = float(ks_2samp(base_df[column], current_df[column]).pvalue)
                drifted = p_value < threshold
                drift_found = drift_found or drifted
                report[column] = {"p_value": p_value, "drift_status": bool(drifted)}
            write_yaml_file(self.data_validation_config.drift_report_file_path, report)
            return drift_found
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            cfg = self.data_validation_config
            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            is_valid = all([
                self.validate_columns(train_df),
                self.validate_columns(test_df),
                self.validate_numeric_columns(train_df),
                self.validate_numeric_columns(test_df),
            ])

            if not is_valid:
                os.makedirs(cfg.invalid_data_dir, exist_ok=True)
                train_df.to_csv(cfg.invalid_train_file_path, index=False, header=True)
                test_df.to_csv(cfg.invalid_test_file_path, index=False, header=True)
                return DataValidationArtifact(
                    validation_status=False,
                    valid_train_file_path=None, valid_test_file_path=None,
                    invalid_train_file_path=cfg.invalid_train_file_path,
                    invalid_test_file_path=cfg.invalid_test_file_path,
                    drift_report_file_path=cfg.drift_report_file_path,
                )

            drift = self.detect_dataset_drift(train_df, test_df)
            if drift:
                logging.warning(f"Drift found between train and test - see {cfg.drift_report_file_path}")

            os.makedirs(cfg.valid_data_dir, exist_ok=True)
            train_df.to_csv(cfg.valid_train_file_path, index=False, header=True)
            test_df.to_csv(cfg.valid_test_file_path, index=False, header=True)

            return DataValidationArtifact(
                validation_status=True,
                valid_train_file_path=cfg.valid_train_file_path,
                valid_test_file_path=cfg.valid_test_file_path,
                invalid_train_file_path=None, invalid_test_file_path=None,
                drift_report_file_path=cfg.drift_report_file_path,
            )
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
