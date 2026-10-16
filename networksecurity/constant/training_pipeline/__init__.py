"""
Common constants for the training pipeline.
Every path here is RELATIVE to the project root, so always run commands
from the project root folder.
"""
import os

# ----------------------------------------------------------------- general
TARGET_COLUMN: str = "Result"
PIPELINE_NAME: str = "NetworkSecurity"
ARTIFACT_DIR: str = "Artifacts"
FILE_NAME: str = "phishingData.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

SCHEMA_FILE_PATH: str = os.path.join("data_schema", "schema.yaml")

# In the raw data: Result = -1 means PHISHING, Result = 1 means LEGITIMATE.
# After transformation we use 1 = phishing (the "positive" class), 0 = legitimate,
# so precision / recall / F1 describe how well we catch phishing.
RAW_PHISHING_LABEL: int = -1
LABEL_NAMES: dict = {0: "legitimate", 1: "phishing"}

# ----------------------------------------------------------------- final model
SAVED_MODEL_DIR: str = "final_model"
MODEL_FILE_NAME: str = "model.pkl"
PREPROCESSOR_FILE_NAME: str = "preprocessor.pkl"
PREDICTION_OUTPUT_DIR: str = "prediction_output"

# ----------------------------------------------------------------- environment
MONGO_URL_ENV_VAR: str = "MONGO_DB_URL"
TRAINING_BUCKET_ENV_VAR: str = "TRAINING_BUCKET_NAME"   # optional (S3 sync)

# ----------------------------------------------------------------- data ingestion
DATA_INGESTION_COLLECTION_NAME: str = "NetworkData"
DATA_INGESTION_DATABASE_NAME: str = "HarshMaurya"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2
# used only when MONGO_DB_URL is not set
DATA_INGESTION_LOCAL_CSV_PATH: str = os.path.join("Network_Data", FILE_NAME)

# ----------------------------------------------------------------- data validation
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"
DATA_VALIDATION_DRIFT_THRESHOLD: float = 0.05

# ----------------------------------------------------------------- data transformation
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"
PREPROCESSING_OBJECT_FILE_NAME: str = "preprocessing.pkl"

# KNN imputer fills missing (NaN) values
DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {
    "missing_values": float("nan"),
    "n_neighbors": 3,
    "weights": "uniform",
}
DATA_TRANSFORMATION_TRAIN_FILE_PATH: str = "train.npy"
DATA_TRANSFORMATION_TEST_FILE_PATH: str = "test.npy"

# ----------------------------------------------------------------- model trainer
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6                    # minimum test F1 allowed
MODEL_TRAINER_OVERFITTING_THRESHOLD: float = 0.05            # max (train F1 - test F1) before warning
MODEL_TRAINER_SEARCH_ITERATIONS: int = 8                     # random-search samples per model
MODEL_TRAINER_CV_FOLDS: int = 3
MODEL_TRAINER_RANDOM_STATE: int = 42
