
from networksecurity.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
)

from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation


def main():
    try:
        # Step 1: Initialize the training pipeline configuration
        training_pipeline_config = TrainingPipelineConfig()

        # Step 2: Configure data ingestion
        data_ingestion_config = DataIngestionConfig(
            training_pipeline_config=training_pipeline_config
        )

        # Step 3: Initialize data ingestion
        data_ingestion = DataIngestion(
            data_ingestion_config=data_ingestion_config
        )

        # Step 4: Execute data ingestion
        data_ingestion_artifact = (
            data_ingestion.initiate_data_ingestion()
        )

        print("\nData ingestion completed successfully!")
        print(data_ingestion_artifact)

        # Step 5: Configure data validation
        data_validation_config = DataValidationConfig(
            training_pipeline_config=training_pipeline_config
        )

        # Step 6: Initialize data validation
        data_validation = DataValidation(
            data_validation_config=data_validation_config,
            data_ingestion_artifact=data_ingestion_artifact,
        )

        # Step 7: Execute data validation
        data_validation_artifact = (
            data_validation.initiate_data_validation()
        )

        print("\nData validation completed successfully!")
        print(data_validation_artifact)

    except Exception as e:
        print(f"Error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
