
import os
import sys
from pathlib import Path

import certifi
import pandas as pd
import pymongo
from dotenv import load_dotenv

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


# Project root: folder containing push_data.py
PROJECT_ROOT = Path(__file__).resolve().parent

# Always load .env from the project root
load_dotenv(PROJECT_ROOT / ".env")

MONGO_DB_URL = os.getenv("MONGODB_URL")
print(MONGO_DB_URL)

class NetworkDataExtract:

    def csv_to_json_convertor(self, file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)

            # Convert each CSV row into a MongoDB document
            records = data.to_dict(orient="records")
            return records

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def insert_data_mongodb(self, records, database, collection):
        client = None

        try:
            if not MONGO_DB_URL:
                raise ValueError(
                    "MONGODB_URL is missing from the project root .env file"
                )

            if not records:
                logging.warning("No records found to insert.")
                return 0

            client = pymongo.MongoClient(
                MONGO_DB_URL,
                tlsCAFile=certifi.where(),
                serverSelectionTimeoutMS=10000,
            )

            # Verify connection before inserting
            client.admin.command("ping")
            logging.info("MongoDB Atlas connection successful.")

            db = client[database]
            mongo_collection = db[collection]

            result = mongo_collection.insert_many(records)

            inserted_count = len(result.inserted_ids)
            logging.info("Inserted %s records.", inserted_count)

            return inserted_count

        except Exception as e:
            raise NetworkSecurityException(e, sys)

        finally:
            if client is not None:
                client.close()


if __name__ == "__main__":
    try:
        FILE_PATH = PROJECT_ROOT / "Network_Data" / "phisingData.csv"
        DATABASE = "HarshMaurya"
        COLLECTION = "NetworkData"

        if not FILE_PATH.exists():
            raise FileNotFoundError(
                f"CSV file not found: {FILE_PATH}"
            )

        network_obj = NetworkDataExtract()

        records = network_obj.csv_to_json_convertor(FILE_PATH)
        print(f"Total CSV records: {len(records)}")

        inserted_count = network_obj.insert_data_mongodb(
            records,
            DATABASE,
            COLLECTION,
        )

        print(f"Successfully inserted {inserted_count} records.")

    except Exception as e:
        logging.exception("Data upload failed.")
        print(f"Upload failed: {e}")
