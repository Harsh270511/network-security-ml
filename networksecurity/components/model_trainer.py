
import os
import sys

import mlflow
import mlflow.sklearn
import numpy as np

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

    def track_mlflow(self, model, metrics, run_name: str):
        """Log a trained model and its evaluation metrics to MLflow."""
        try:
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)

            mlflow.set_experiment("NetworkSecurity_Experiment")

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

                # Explicitly trust the sklearn Tree type used by
                # decision trees and tree-based ensemble estimators.
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    skops_trusted_types=[
                        "sklearn.tree._tree.Tree"
                    ],
                )

            logging.info(
                f"MLflow run completed successfully: {run_name}"
            )

        except Exception as e:
            logging.exception(
                f"MLflow logging failed for run '{run_name}'"
            )
            raise NetworkSecurityException(e, sys) from e

    def train_model(self, X_train, y_train, X_test, y_test):
        try:
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

            logging.info("Starting model evaluation and tuning.")

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

            # Select the estimator with the highest cross-validation F1.
            best_model_name = max(
                model_report,
                key=lambda name: model_report[name]["cv_f1"],
            )

            best_model_score = model_report[best_model_name]["cv_f1"]
            best_model = model_report[best_model_name]["estimator"]

            logging.info(
                f"Best model: {best_model_name}, "
                f"CV F1 score: {best_model_score:.4f}"
            )

            # Evaluate on the training dataset.
            y_train_pred = best_model.predict(X_train)

            train_metrics = get_classification_score(
                y_true=y_train,
                y_pred=y_train_pred,
            )

            # Evaluate on the held-out test dataset.
            y_test_pred = best_model.predict(X_test)

            test_metrics = get_classification_score(
                y_true=y_test,
                y_pred=y_test_pred,
            )

            logging.info(f"Training metrics: {train_metrics}")
            logging.info(f"Testing metrics: {test_metrics}")

            # Log separate MLflow runs for training and test metrics.
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

            # Load the preprocessing object created during transformation.
            preprocessor = load_object(
                file_path=(
                    self.data_transformation_artifact
                    .transformed_object_file_path
                )
            )

            # Combine preprocessing and the trained estimator.
            network_model = NetworkModel(
                preprocessor=preprocessor,
                model=best_model,
            )

            model_path = (
                self.model_trainer_config.trained_model_file_path
            )

            model_directory = os.path.dirname(model_path)

            if model_directory:
                os.makedirs(
                    model_directory,
                    exist_ok=True,
                )

            # Save the complete prediction pipeline.
            save_object(
                file_path=model_path,
                obj=network_model,
            )

            artifact = ModelTrainerArtifact(
                trained_model_file_path=model_path,
                train_metric_artifact=train_metrics,
                test_metric_artifact=test_metrics,
            )

            logging.info(
                f"Model trainer completed successfully: {artifact}"
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
                    "Transformed train and test arrays must be 2D."
                )

            if train_array.shape[1] < 2 or test_array.shape[1] < 2:
                raise ValueError(
                    "Transformed arrays must contain features "
                    "and a target column."
                )

            # Last column is the target; all preceding columns are features.
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            logging.info(
                f"Training data shape: {X_train.shape}; "
                f"target shape: {y_train.shape}"
            )
            logging.info(
                f"Testing data shape: {X_test.shape}; "
                f"target shape: {y_test.shape}"
            )

            return self.train_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )

        except Exception as e:
            logging.exception(
                "Could not initiate model training."
            )
            raise NetworkSecurityException(e, sys) from e
