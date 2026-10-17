
import os
import sys

import numpy as np
import pandas as pd
import pymongo
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact

from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_FILE, override=False)

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

print("Project root:", PROJECT_ROOT)
print(".env exists:", ENV_FILE.is_file())
print("MONGO_DB_URL loaded:", bool(MONGO_DB_URL))


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config

            if not MONGO_DB_URL:
                raise ValueError(
                    "MONGO_DB_URL is missing. Check your .env file."
                )

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def export_collection_as_dataframe(self) -> pd.DataFrame:
        """Read data from MongoDB and return it as a DataFrame."""
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name

            self.mongo_client = pymongo.MongoClient(
                MONGO_DB_URL,
                serverSelectionTimeoutMS=10000,
            )

            # Verify MongoDB connectivity before querying.
            self.mongo_client.admin.command("ping")

            collection = self.mongo_client[database_name][collection_name]
            df = pd.DataFrame(list(collection.find()))

            # Remove MongoDB's generated ID column if present.
            df = df.drop(columns=["_id"], errors="ignore")

            df.replace({"na": np.nan}, inplace=True)

            logging.info(
                f"Retrieved {len(df)} records from "
                f"{database_name}.{collection_name}"
            )

            if df.empty:
                raise ValueError(
                    f"No records found in {database_name}.{collection_name}"
                )

            return df

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

        finally:
            if hasattr(self, "mongo_client"):
                self.mongo_client.close()

    def export_data_into_feature_store(
        self, dataframe: pd.DataFrame
    ) -> pd.DataFrame:
        try:
            feature_store_file_path = (
                self.data_ingestion_config.feature_store_file_path
            )

            dir_path = os.path.dirname(feature_store_file_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)

            dataframe.to_csv(
                feature_store_file_path,
                index=False,
                header=True,
            )

            logging.info(
                f"Feature store saved at {feature_store_file_path}"
            )

            return dataframe

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def split_data_as_train_test(
        self, dataframe: pd.DataFrame
    ) -> None:
        try:
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42,
            )

            logging.info("Performed train-test split.")

            train_path = self.data_ingestion_config.training_file_path
            test_path = self.data_ingestion_config.testing_file_path

            for file_path in (train_path, test_path):
                dir_path = os.path.dirname(file_path)
                if dir_path:
                    os.makedirs(dir_path, exist_ok=True)

            train_set.to_csv(train_path, index=False, header=True)
            test_set.to_csv(test_path, index=False, header=True)

            logging.info(f"Training data saved at {train_path}")
            logging.info(f"Testing data saved at {test_path}")

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            dataframe = self.export_collection_as_dataframe()

            self.export_data_into_feature_store(dataframe)

            self.split_data_as_train_test(dataframe)

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )

            logging.info("Data ingestion completed successfully.")

            return data_ingestion_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
