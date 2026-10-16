import os
import sys

import certifi
import numpy as np
import pandas as pd
import pymongo
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from networksecurity.constant.training_pipeline import MONGO_URL_ENV_VAR, TARGET_COLUMN
from networksecurity.entity.artifact_entity import DataIngestionArtifact
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

load_dotenv()


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config

    def export_collection_as_dataframe(self) -> pd.DataFrame:
        """
        Read the data from MongoDB (when MONGO_DB_URL is set in .env).
        Otherwise fall back to the local CSV so the project still runs offline.
        """
        try:
            mongo_url = os.getenv(MONGO_URL_ENV_VAR, "").strip()
            if mongo_url:
                logging.info("Reading data from MongoDB")
                client = pymongo.MongoClient(
                    mongo_url, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=10000
                )
                try:
                    collection = client[self.data_ingestion_config.database_name][
                        self.data_ingestion_config.collection_name
                    ]
                    df = pd.DataFrame(list(collection.find()))
                finally:
                    client.close()
                if df.empty:
                    raise ValueError(
                        "MongoDB collection is empty. Run `python scripts/push_data.py` first."
                    )
                df = df.drop(columns=["_id"], errors="ignore")
            else:
                logging.warning(
                    f"{MONGO_URL_ENV_VAR} is not set - reading the local CSV "
                    f"{self.data_ingestion_config.local_csv_path} instead"
                )
                df = pd.read_csv(self.data_ingestion_config.local_csv_path)

            df = df.replace({"na": np.nan})
            before = len(df)
            df = df.drop_duplicates().reset_index(drop=True)
            logging.info(f"Dropped {before - len(df)} duplicate rows ({len(df)} remain)")
            return df
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def export_data_into_feature_store(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        try:
            path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(os.path.dirname(path), exist_ok=True)
            dataframe.to_csv(path, index=False, header=True)
            return dataframe
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def split_data_as_train_test(self, dataframe: pd.DataFrame) -> None:
        try:
            stratify = dataframe[TARGET_COLUMN] if TARGET_COLUMN in dataframe.columns else None
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42,
                stratify=stratify,
            )
            os.makedirs(os.path.dirname(self.data_ingestion_config.training_file_path), exist_ok=True)
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
            logging.info(f"Train/test split done: {len(train_set)} / {len(test_set)} rows")
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            dataframe = self.export_collection_as_dataframe()
            dataframe = self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)
            return DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
 