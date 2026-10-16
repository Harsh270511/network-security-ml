
from networksecurity.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
)
from networksecurity.components.data_ingestion import DataIngestion


def main():
    try:
        # Step 1: Initialize the training pipeline configuration
        training_pipeline_config = TrainingPipelineConfig()

        # Step 2: Initialize the data ingestion configuration
        data_ingestion_config = DataIngestionConfig(
            training_pipeline_config=training_pipeline_config
        )

        # Step 3: Create the DataIngestion object
        data_ingestion = DataIngestion(
            data_ingestion_config=data_ingestion_config
        )

        # Step 4: Run data ingestion
        data_ingestion_artifact = (
            data_ingestion.initiate_data_ingestion()
        )

        print("\nData ingestion completed successfully!")
        print(data_ingestion_artifact)

    except Exception as e:
        print(f"Error occurred: {e}")
        raise e


if __name__ == "__main__":
    main()
