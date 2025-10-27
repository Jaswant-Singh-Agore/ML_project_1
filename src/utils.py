import os
import sys
import dill
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException


# Save Python object safely (used for model + preprocessor)
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)  # dill > pickle for better compatibility

    except Exception as e:
        raise CustomException(e, sys)


# Load Python object safely (used in prediction pipeline)
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


# Evaluate models using GridSearchCV and return trained models + report
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        trained_models = {}

        for model_name, model in models.items():
            parameters = param.get(model_name, {})

            # Perform hyperparameter tuning
            gs = GridSearchCV(model, parameters, cv=3, n_jobs=-1, verbose=0)
            gs.fit(X_train, y_train)

            # Best model from GridSearchCV
            best_model = gs.best_estimator_
            best_model.fit(X_train, y_train)

            # Evaluate test performance
            y_test_pred = best_model.predict(X_test)
            test_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_score
            trained_models[model_name] = best_model

            print(f">>> {model_name}: R² = {test_score:.4f}")

        # Return both report and trained model dictionary
        return report, trained_models

    except Exception as e:
        raise CustomException(e, sys)
