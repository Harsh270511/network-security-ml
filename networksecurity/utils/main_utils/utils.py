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


def evaluate_models(X_train, y_train, models: dict, params: dict,
                    cv: int = 3, n_iter: int = 8, random_state: int = 42) -> dict:
    """
    Tune every model with cross-validation on the TRAINING data only
    (the test set is never used to choose a model) and return

        {model_name: {"cv_f1": float, "estimator": fitted_best_estimator}}
    """
    try:
        report = {}
        for name, model in models.items():
            grid = params.get(name) or {}
            logging.info(f"Tuning {name} ...")
            if grid:
                search = RandomizedSearchCV(
                    model, grid, n_iter=n_iter, cv=cv, scoring="f1",
                    random_state=random_state, n_jobs=-1,
                )
                search.fit(X_train, y_train)
                best_estimator, cv_f1 = search.best_estimator_, float(search.best_score_)
            else:
                cv_f1 = float(np.mean(cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")))
                best_estimator = model.fit(X_train, y_train)
            logging.info(f"{name}: cross-validated F1 = {cv_f1:.4f}")
            report[name] = {"cv_f1": cv_f1, "estimator": best_estimator}
        return report
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
