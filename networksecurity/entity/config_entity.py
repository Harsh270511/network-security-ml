import os
from datetime import datetime
from typing import Optional

from networksecurity.constant import training_pipeline as tp


class TrainingPipelineConfig:
    def __init__(self, timestamp: Optional[datetime] = None):
        # NOTE: the timestamp is created on every call (not once at import time),
        # so each /train request gets its own Artifacts/<timestamp> folder.
        timestamp = (timestamp or datetime.now()).strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name: str = tp.PIPELINE_NAME
        self.artifact_name: str = tp.ARTIFACT_DIR
        self.artifact_dir: str = os.path.join(self.artifact_name, timestamp)
        self.model_dir: str = tp.SAVED_MODEL_DIR
        self.timestamp: str = timestamp


class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, tp.DATA_INGESTION_DIR_NAME
        )
        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir, tp.DATA_INGESTION_FEATURE_STORE_DIR, tp.FILE_NAME
        )
        self.training_file_path: str = os.path.join(
            self.data_ingestion_dir, tp.DATA_INGESTION_INGESTED_DIR, tp.TRAIN_FILE_NAME
        )
        self.testing_file_path: str = os.path.join(
            self.data_ingestion_dir, tp.DATA_INGESTION_INGESTED_DIR, tp.TEST_FILE_NAME
        )
        self.train_test_split_ratio: float = tp.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        self.collection_name: str = tp.DATA_INGESTION_COLLECTION_NAME
        self.database_name: str = tp.DATA_INGESTION_DATABASE_NAME
        self.local_csv_path: str = tp.DATA_INGESTION_LOCAL_CSV_PATH


class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_validation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, tp.DATA_VALIDATION_DIR_NAME
        )
        self.valid_data_dir: str = os.path.join(self.data_validation_dir, tp.DATA_VALIDATION_VALID_DIR)
        self.invalid_data_dir: str = os.path.join(self.data_validation_dir, tp.DATA_VALIDATION_INVALID_DIR)
        self.valid_train_file_path: str = os.path.join(self.valid_data_dir, tp.TRAIN_FILE_NAME)
        self.valid_test_file_path: str = os.path.join(self.valid_data_dir, tp.TEST_FILE_NAME)
        self.invalid_train_file_path: str = os.path.join(self.invalid_data_dir, tp.TRAIN_FILE_NAME)
        self.invalid_test_file_path: str = os.path.join(self.invalid_data_dir, tp.TEST_FILE_NAME)
        self.drift_report_file_path: str = os.path.join(
            self.data_validation_dir,
            tp.DATA_VALIDATION_DRIFT_REPORT_DIR,
            tp.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,
        )
        self.drift_threshold: float = tp.DATA_VALIDATION_DRIFT_THRESHOLD


class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_transformation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, tp.DATA_TRANSFORMATION_DIR_NAME
        )
        self.transformed_train_file_path: str = os.path.join(
            self.data_transformation_dir,
            tp.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            tp.TRAIN_FILE_NAME.replace("csv", "npy"),
        )
        self.transformed_test_file_path: str = os.path.join(
            self.data_transformation_dir,
            tp.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            tp.TEST_FILE_NAME.replace("csv", "npy"),
        )
        self.transformed_object_file_path: str = os.path.join(
            self.data_transformation_dir,
            tp.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            tp.PREPROCESSING_OBJECT_FILE_NAME,
        )


class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_trainer_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, tp.MODEL_TRAINER_DIR_NAME
        )
        self.trained_model_file_path: str = os.path.join(
            self.model_trainer_dir,
            tp.MODEL_TRAINER_TRAINED_MODEL_DIR,
            tp.MODEL_TRAINER_TRAINED_MODEL_NAME,
        )
        self.final_model_dir: str = training_pipeline_config.model_dir
        self.expected_score: float = tp.MODEL_TRAINER_EXPECTED_SCORE
        self.overfitting_threshold: float = tp.MODEL_TRAINER_OVERFITTING_THRESHOLD
 