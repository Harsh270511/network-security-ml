import sys

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from networksecurity.constant.training_pipeline import (
    DATA_TRANSFORMATION_IMPUTER_PARAMS,
    RAW_PHISHING_LABEL,
    TARGET_COLUMN,
)
from networksecurity.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.utils.main_utils.utils import save_numpy_array_data, save_object


class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        self.data_validation_artifact = data_validation_artifact
        self.data_transformation_config = data_transformation_config

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    @staticmethod
    def get_data_transformer_object() -> Pipeline:
        """A pipeline with a KNNImputer (fills NaN values using the 3 nearest rows)."""
        try:
            imputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            return Pipeline([("imputer", imputer)])
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    @staticmethod
    def encode_target(target: pd.Series) -> np.ndarray:
        """Raw data: -1 = phishing, 1 = legitimate  ->  1 = phishing, 0 = legitimate."""
        return (target == RAW_PHISHING_LABEL).astype(int).to_numpy()

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            if not self.data_validation_artifact.validation_status:
                raise ValueError("Data validation failed - refusing to transform invalid data")

            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)

            x_train = train_df.drop(columns=[TARGET_COLUMN])
            y_train = self.encode_target(train_df[TARGET_COLUMN])
            x_test = test_df.drop(columns=[TARGET_COLUMN])
            y_test = self.encode_target(test_df[TARGET_COLUMN])

            preprocessor = self.get_data_transformer_object().fit(x_train)   # fit on TRAIN only
            x_train_t = preprocessor.transform(x_train)
            x_test_t = preprocessor.transform(x_test)

            train_arr = np.c_[x_train_t, y_train]
            test_arr = np.c_[x_test_t, y_test]

            cfg = self.data_transformation_config
            save_numpy_array_data(cfg.transformed_train_file_path, train_arr)
            save_numpy_array_data(cfg.transformed_test_file_path, test_arr)
            save_object(cfg.transformed_object_file_path, preprocessor)
            logging.info("Data transformation completed")

            return DataTransformationArtifact(
                transformed_object_file_path=cfg.transformed_object_file_path,
                transformed_train_file_path=cfg.transformed_train_file_path,
                transformed_test_file_path=cfg.transformed_test_file_path,
            )
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
