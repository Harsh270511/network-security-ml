from networksecurity.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
)

from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation


def main():
    try:
        # Step 1: Initialize training pipeline configuration
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

        # Step 8: Configure data transformation
        data_transformation_config = DataTransformationConfig(
            training_pipeline_config=training_pipeline_config
        )

        # Step 9: Initialize data transformation
        data_transformation = DataTransformation(
            data_transformation_config=data_transformation_config,
            data_validation_artifact=data_validation_artifact,
        )

        # Step 10: Execute data transformation
        data_transformation_artifact = (
            data_transformation.initiate_data_transformation()
        )

        print("\nData transformation completed successfully!")
        print(data_transformation_artifact)

    except Exception as e:
        print(f"Error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
