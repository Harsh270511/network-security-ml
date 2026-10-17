
import sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer

from networksecurity.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)

from networksecurity.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)

from networksecurity.constant.training_pipeline import TRAINING_BUCKET_NAME
from networksecurity.cloud.s3_syncer import S3Sync


class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        self.s3_sync = S3Sync()

    def start_data_ingestion(self):
        try:
            logging.info("Starting data ingestion")

            data_ingestion_config = DataIngestionConfig(
                training_pipeline_config=self.training_pipeline_config
            )

            data_ingestion = DataIngestion(
                data_ingestion_config=data_ingestion_config
            )

            artifact = data_ingestion.initiate_data_ingestion()

            logging.info("Data ingestion completed: %s", artifact)
            return artifact

        except Exception as e:
            logging.exception("Data ingestion failed")
            raise NetworkSecurityException(e, sys) from e

    def start_data_validation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
    ):
        try:
            logging.info("Starting data validation")

            data_validation_config = DataValidationConfig(
                training_pipeline_config=self.training_pipeline_config
            )

            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=data_validation_config,
            )

            artifact = data_validation.initiate_data_validation()

            logging.info("Data validation completed: %s", artifact)
            return artifact

        except Exception as e:
            logging.exception("Data validation failed")
            raise NetworkSecurityException(e, sys) from e

    def start_data_transformation(
        self,
        data_validation_artifact: DataValidationArtifact,
    ):
        try:
            logging.info("Starting data transformation")

            data_transformation_config = DataTransformationConfig(
                training_pipeline_config=self.training_pipeline_config
            )

            data_transformation = DataTransformation(
                data_validation_artifact=data_validation_artifact,
                data_transformation_config=data_transformation_config,
            )

            artifact = data_transformation.initiate_data_transformation()

            logging.info("Data transformation completed: %s", artifact)
            return artifact

        except Exception as e:
            logging.exception("Data transformation failed")
            raise NetworkSecurityException(e, sys) from e

    def start_model_trainer(
        self,
        data_transformation_artifact: DataTransformationArtifact,
    ) -> ModelTrainerArtifact:
        try:
            logging.info("Starting model training")

            model_trainer_config = ModelTrainerConfig(
                training_pipeline_config=self.training_pipeline_config
            )

            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=model_trainer_config,
            )

            artifact = model_trainer.initiate_model_trainer()

            logging.info("Model training completed: %s", artifact)
            return artifact

        except Exception as e:
            logging.exception("Model training failed")
            raise NetworkSecurityException(e, sys) from e

    def sync_artifact_dir_to_s3(self):
        try:
            bucket_url = (
                f"s3://{TRAINING_BUCKET_NAME}/artifact/"
                f"{self.training_pipeline_config.timestamp}"
            )

            logging.info("Syncing artifacts to %s", bucket_url)

            self.s3_sync.sync_folder_to_s3(
                folder=self.training_pipeline_config.artifact_dir,
                aws_bucket_url=bucket_url,
            )

        except Exception as e:
            logging.exception("Artifact S3 sync failed")
            raise NetworkSecurityException(e, sys) from e

    def sync_saved_model_dir_to_s3(self):
        try:
            bucket_url = (
                f"s3://{TRAINING_BUCKET_NAME}/final_model/"
                f"{self.training_pipeline_config.timestamp}"
            )

            logging.info("Syncing final model to %s", bucket_url)

            self.s3_sync.sync_folder_to_s3(
                folder=self.training_pipeline_config.model_dir,
                aws_bucket_url=bucket_url,
            )

        except Exception as e:
            logging.exception("Final model S3 sync failed")
            raise NetworkSecurityException(e, sys) from e

    def run_pipeline(self):
        try:
            logging.info(
                "Pipeline started. Artifact directory: %s",
                self.training_pipeline_config.artifact_dir,
            )

            data_ingestion_artifact = self.start_data_ingestion()

            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )

            data_transformation_artifact = self.start_data_transformation(
                data_validation_artifact=data_validation_artifact
            )

            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact=data_transformation_artifact
            )

            logging.info("All training stages completed.")

            self.sync_artifact_dir_to_s3()
            self.sync_saved_model_dir_to_s3()

            logging.info("Training pipeline completed successfully.")
            return model_trainer_artifact

        except Exception as e:
            logging.exception("Training pipeline failed.")
            raise NetworkSecurityException(e, sys) from e
