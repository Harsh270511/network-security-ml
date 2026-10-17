
import os
import sys

import mlflow
import mlflow.sklearn
from dotenv import load_dotenv

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array_data,
    evaluate_models,
)
from networksecurity.utils.ml_utils.metric.classification_metric import (
    get_classification_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)


# Load credentials and configuration from the project-root .env file.
load_dotenv()

DAGSHUB_REPO_OWNER = "hm0093173"
DAGSHUB_REPO_NAME = "network-security-ml"

DAGSHUB_MLFLOW_URI = (
    f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/"
    f"{DAGSHUB_REPO_NAME}.mlflow"
)


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def configure_mlflow(self):
        """Configure MLflow to use this project's DagsHub repository."""
        try:
            username = os.getenv("MLFLOW_TRACKING_USERNAME")
            token = os.getenv("MLFLOW_TRACKING_PASSWORD")

            if not username or not token:
                raise EnvironmentError(
                    "DagsHub credentials are missing. Set "
                    "MLFLOW_TRACKING_USERNAME and "
                    "MLFLOW_TRACKING_PASSWORD in your .env file."
                )

            # Use your own DagsHub repository, not Krish Naik's.
            mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
            mlflow.set_registry_uri(DAGSHUB_MLFLOW_URI)

            mlflow.set_experiment("NetworkSecurity_Experiment")

            logging.info(
                f"MLflow tracking URI configured: {DAGSHUB_MLFLOW_URI}"
            )

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def track_mlflow(self, model, metrics, run_name: str):
        """Log a fitted model and its metrics to DagsHub."""
        try:
            with mlflow.start_run(run_name=run_name):
                mlflow.log_metric(
                    "f1_score",
                    float(metrics.f1_score),
                )
                mlflow.log_metric(
                    "precision",
                    float(metrics.precision_score),
                )
                mlflow.log_metric(
                    "recall_score",
                    float(metrics.recall_score),
                )

                mlflow.log_param("model_type", type(model).__name__)

                # This is appropriate for a model trained in this project.
                # It allows skops to serialize sklearn decision-tree objects.
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    skops_trusted_types=[
                        "sklearn.tree._tree.Tree"
                    ],
                )

            logging.info(f"Logged MLflow run: {run_name}")

        except Exception as e:
            logging.exception(
                f"Failed to log MLflow run: {run_name}"
            )
            raise NetworkSecurityException(e, sys) from e

    def train_model(self, X_train, y_train, X_test, y_test):
        try:
            self.configure_mlflow()

            models = {
                "Random Forest": RandomForestClassifier(
                    random_state=42,
                ),
                "Decision Tree": DecisionTreeClassifier(
                    random_state=42,
                ),
                "Gradient Boosting": GradientBoostingClassifier(
                    random_state=42,
                ),
                "Logistic Regression": LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
                "AdaBoost": AdaBoostClassifier(
                    random_state=42,
                ),
                "KNeighbors": KNeighborsClassifier(),
            }

            params = {
                "Decision Tree": {
                    "criterion": [
                        "gini",
                        "entropy",
                        "log_loss",
                    ],
                },
                "Random Forest": {
                    "n_estimators": [8, 16, 32, 128],
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.05],
                    "subsample": [0.6, 0.8, 0.9],
                    "n_estimators": [8, 16, 32, 64],
                },
                "Logistic Regression": {
                    "C": [0.1, 1.0, 10.0],
                },
                "AdaBoost": {
                    "learning_rate": [0.1, 0.01, 1.0],
                    "n_estimators": [8, 16, 32, 64],
                },
                "KNeighbors": {
                    "n_neighbors": [3, 5, 7],
                },
            }

            logging.info("Starting model evaluation.")

            # Match the evaluate_models signature in utils.py:
            # it returns a dictionary with cv_f1 and estimator per model.
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                models=models,
                params=params,
            )

            if not model_report:
                raise ValueError(
                    "Model evaluation returned no results."
                )

            best_model_name = max(
                model_report,
                key=lambda name: model_report[name]["cv_f1"],
            )

            best_model_score = model_report[best_model_name]["cv_f1"]
            best_model = model_report[best_model_name]["estimator"]

            logging.info(
                f"Selected model: {best_model_name}; "
                f"cross-validation F1: {best_model_score:.4f}"
            )

            # Evaluate the fitted best estimator on training data.
            y_train_pred = best_model.predict(X_train)
            train_metrics = get_classification_score(
                y_true=y_train,
                y_pred=y_train_pred,
            )

            # Evaluate on the held-out test data.
            y_test_pred = best_model.predict(X_test)
            test_metrics = get_classification_score(
                y_true=y_test,
                y_pred=y_test_pred,
            )

            logging.info(f"Training metrics: {train_metrics}")
            logging.info(f"Testing metrics: {test_metrics}")

            # Log separate training and test runs to DagsHub.
            self.track_mlflow(
                model=best_model,
                metrics=train_metrics,
                run_name=f"{best_model_name}_train",
            )

            self.track_mlflow(
                model=best_model,
                metrics=test_metrics,
                run_name=f"{best_model_name}_test",
            )

            # Load the fitted preprocessing pipeline.
            preprocessor = load_object(
                file_path=(
                    self.data_transformation_artifact
                    .transformed_object_file_path
                )
            )

            network_model = NetworkModel(
                preprocessor=preprocessor,
                model=best_model,
            )

            model_path = (
                self.model_trainer_config.trained_model_file_path
            )

            model_directory = os.path.dirname(model_path)
            if model_directory:
                os.makedirs(model_directory, exist_ok=True)

            # Save the complete preprocessing + model instance.
            save_object(
                file_path=model_path,
                obj=network_model,
            )

            # Keep a convenient copy at final_model/model.pkl.
            save_object(
                file_path=os.path.join(
                    "final_model",
                    "model.pkl",
                ),
                obj=network_model,
            )

            artifact = ModelTrainerArtifact(
                trained_model_file_path=model_path,
                train_metric_artifact=train_metrics,
                test_metric_artifact=test_metrics,
            )

            logging.info(
                f"Model trainer completed: {artifact}"
            )

            return artifact

        except Exception as e:
            logging.exception("Model training failed.")
            raise NetworkSecurityException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = (
                self.data_transformation_artifact
                .transformed_train_file_path
            )
            test_file_path = (
                self.data_transformation_artifact
                .transformed_test_file_path
            )

            train_array = load_numpy_array_data(train_file_path)
            test_array = load_numpy_array_data(test_file_path)

            if train_array.ndim != 2 or test_array.ndim != 2:
                raise ValueError(
                    "Train and test arrays must both be two-dimensional."
                )

            if train_array.shape[1] < 2 or test_array.shape[1] < 2:
                raise ValueError(
                    "Each array must contain features and a target column."
                )

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            logging.info(
                f"Train features: {X_train.shape}; "
                f"test features: {X_test.shape}"
            )

            return self.train_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )

        except Exception as e:
            logging.exception("Could not initiate model training.")
            raise NetworkSecurityException(e, sys) from e
