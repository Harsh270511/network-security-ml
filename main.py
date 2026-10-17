
import sys

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

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


def main():
    try:
        # Step 1: Initialize pipeline configuration
        logging.info("Initializing training pipeline configuration")
        training_pipeline_config = TrainingPipelineConfig()

        # Step 2: Data Ingestion
        logging.info("Starting data ingestion")
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)

        data_ingestion_artifact = (
            data_ingestion.initiate_data_ingestion()
        )
        logging.info("Data ingestion completed successfully")
        print(data_ingestion_artifact)

        # Step 3: Data Validation
        logging.info("Starting data validation")
        data_validation_config = DataValidationConfig(
            training_pipeline_config
        )
        data_validation = DataValidation(
            data_ingestion_artifact,
            data_validation_config,
        )

        data_validation_artifact = (
            data_validation.initiate_data_validation()
        )
        logging.info("Data validation completed")
        print(data_validation_artifact)

        # Stop if validation failed
        if not data_validation_artifact.validation_status:
            raise ValueError(
                "Data validation failed. Check the drift report and schema."
            )

        # Step 4: Data Transformation
        logging.info("Starting data transformation")
        data_transformation_config = DataTransformationConfig(
            training_pipeline_config
        )
        data_transformation = DataTransformation(
            data_validation_artifact,
            data_transformation_config,
        )

        data_transformation_artifact = (
            data_transformation.initiate_data_transformation()
        )
        logging.info("Data transformation completed")
        print(data_transformation_artifact)

        # Step 5: Model Training
        logging.info("Starting model training")
        model_trainer_config = ModelTrainerConfig(
            training_pipeline_config
        )
        model_trainer = ModelTrainer(
            model_trainer_config=model_trainer_config,
            data_transformation_artifact=data_transformation_artifact,
        )

        model_trainer_artifact = (
            model_trainer.initiate_model_trainer()
        )
        logging.info("Model training completed successfully")
        print(model_trainer_artifact)

    except Exception as e:
        logging.exception("Training pipeline failed")
        raise NetworkSecurityException(e, sys) from e


if __name__ == "__main__":
    main()
