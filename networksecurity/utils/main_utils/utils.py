import os
import pickle
import sys

import numpy as np
import yaml
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "r", encoding="utf-8") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def save_numpy_array_data(file_path: str, array: np.ndarray) -> None:
    """Save a numpy array to `file_path` (folders are created automatically)."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.ndarray:
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def save_object(file_path: str, obj: object) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Saved object to {file_path}")
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def load_object(file_path: str) -> object:
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist")
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e



from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)


def evaluate_models(
    X_train,
    y_train,
    models: dict,
    params: dict,
    cv: int = 3,
    n_iter: int = 8,
    random_state: int = 42,
) -> dict:
    """
    Tune classification models using training data only.

    Returns:
        {
            model_name: {
                "cv_f1": float,
                "estimator": fitted_best_estimator
            }
        }
    """
    try:
        report = {}

        cv_strategy = StratifiedKFold(
            n_splits=cv,
            shuffle=True,
            random_state=random_state,
        )

        for name, model in models.items():
            model_params = params.get(name) or {}

            logging.info(f"Tuning {name}...")

            if model_params:
                search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=model_params,
                    n_iter=n_iter,
                    cv=cv_strategy,
                    scoring="f1",
                    random_state=random_state,
                    n_jobs=-1,
                    error_score="raise",
                )

                search.fit(X_train, y_train)

                best_estimator = search.best_estimator_
                cv_f1 = float(search.best_score_)

                logging.info(
                    f"{name} best parameters: {search.best_params_}"
                )

            else:
                scores = cross_val_score(
                    model,
                    X_train,
                    y_train,
                    cv=cv_strategy,
                    scoring="f1",
                    n_jobs=-1,
                )

                cv_f1 = float(np.mean(scores))

                # Fit a fresh estimator on the full training data.
                best_estimator = model.fit(X_train, y_train)

            report[name] = {
                "cv_f1": cv_f1,
                "estimator": best_estimator,
            }

            logging.info(
                f"{name}: CV F1 = {cv_f1:.4f}"
            )

        return report

    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
