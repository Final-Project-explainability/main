import os
import joblib
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
import random

from explainability.src.FeatureFilteredXGB import FeatureFilteredXGB

random.seed(42)
np.random.seed(42)


def train_model(X_train, y_train, model_path="gradient_boosting_model.joblib"):
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("Gradient Boosting model loaded from file.")
    else:
        model = GradientBoostingClassifier()
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        print("Gradient Boosting model trained and saved to file!")
    return model


# def train_xgboost(X_train, y_train, model_path="xgboost_model.joblib", tune=True):
#     if os.path.exists(model_path):
#         model = joblib.load(model_path)
#         print("XGBoost model loaded from file.")
#     else:
#         # Set scale_pos_weight based on class imbalance
#         scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
#
#         if tune:
#             # Define parameter grid for hyperparameter tuning
#             param_grid = {
#                 'max_depth': [3, 5, 7],
#                 'learning_rate': [0.01, 0.1, 0.2],
#                 'n_estimators': [100, 200, 300],
#                 'scale_pos_weight': [1, scale_pos_weight],
#                 'min_child_weight': [1, 5, 10],
#                 'subsample': [0.8, 1],
#                 'colsample_bytree': [0.8, 1]
#             }
#
#             xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
#             grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='f1', cv=3, verbose=1, n_jobs=-1)
#             grid_search.fit(X_train, y_train)
#
#             # Use the best model from grid search
#             model = grid_search.best_estimator_
#             print("Best parameters found: ", grid_search.best_params_)
#         else:
#             # Train with default or manually set parameters
#             model = xgb.XGBClassifier(
#                 objective='binary:logistic',
#                 eval_metric='logloss',
#                 max_depth=5,
#                 learning_rate=0.1,
#                 n_estimators=200,
#                 scale_pos_weight=scale_pos_weight,
#                 min_child_weight=5,
#                 subsample=0.8,
#                 colsample_bytree=0.8
#             )
#             model.fit(X_train, y_train)
#
#         # Save the trained model
#         joblib.dump(model, model_path)
#         print("XGBoost model trained and saved to file!")
#
#     return model


# def train_xgboost(X_train, y_train, model_path="xgboost_model.joblib", tune=True): # oded version RandomizedSearchCV
#     if os.path.exists(model_path):
#         model = joblib.load(model_path)
#         print("XGBoost model loaded from file.")
#     else:
#         scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
#
#         if tune:
#             # Expanded parameter grid for improving ROC AUC
#             param_grid = {
#                 'max_depth': [3, 4, 5],
#                 'learning_rate': [0.01, 0.05, 0.1],
#                 'n_estimators': [200, 300],
#                 'scale_pos_weight': [1, scale_pos_weight],
#                 'min_child_weight': [1, 5, 10],
#                 'gamma': [0, 0.1, 0.2],
#                 'subsample': [0.7, 0.8, 0.9],
#                 'colsample_bytree': [0.7, 0.8, 0.9]
#             }
#
#             xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
#             random_search = RandomizedSearchCV(
#                 estimator=xgb_model,
#                 param_distributions=param_grid,
#                 scoring='roc_auc',  # Focus on maximizing ROC AUC
#                 cv=3,
#                 n_iter=10,
#                 verbose=1,
#                 n_jobs=-1
#             )
#             random_search.fit(X_train, y_train)
#
#             model = random_search.best_estimator_
#             print("Best parameters found: ", random_search.best_params_)
#         else:
#             model = xgb.XGBClassifier(
#                 objective='binary:logistic',
#                 eval_metric='logloss',
#                 max_depth=4,
#                 learning_rate=0.05,
#                 n_estimators=300,
#                 scale_pos_weight=scale_pos_weight,
#                 min_child_weight=5,
#                 gamma=0.1,
#                 subsample=0.8,
#                 colsample_bytree=0.8
#             )
#             model.fit(X_train, y_train)
#
#         joblib.dump(model, model_path)
#         print("XGBoost model trained and saved to file!")
#
#     return model

# def train_xgboost(X_train, y_train, model_path="xgboost_model.joblib", tune=True): # changed in 10:41 14.11.24
#     if os.path.exists(model_path):
#         model = joblib.load(model_path)
#         print("XGBoost model loaded from file.")
#     else:
#         scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
#
#         if tune:
#             def objective(trial):
#                 # Define the parameter space for Bayesian Optimization
#                 param = {
#                     'objective': 'binary:logistic',
#                     'eval_metric': 'logloss',
#                     'max_depth': trial.suggest_int('max_depth', 3, 10),
#                     'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
#                     'n_estimators': trial.suggest_int('n_estimators', 100, 500),
#                     'scale_pos_weight': scale_pos_weight,
#                     'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
#                     'gamma': trial.suggest_float('gamma', 0.0001, 1.0, log=True),  # Fixed range for gamma
#                     'subsample': trial.suggest_float('subsample', 0.6, 1.0),
#                     'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
#                 }
#
#                 # Use cross-validation to evaluate the model on training data only
#                 model = xgb.XGBClassifier(**param)
#                 cv_scores = cross_val_score(model, X_train, y_train, scoring='roc_auc', cv=3)
#                 return cv_scores.mean()
#
#             # Optimize the objective function using Optuna
#             study = optuna.create_study(direction="maximize")
#             study.optimize(objective, n_trials=500)
#
#             # Get the best parameters found by Optuna
#             best_params = study.best_params
#             best_params['objective'] = 'binary:logistic'
#             best_params['eval_metric'] = 'logloss'
#             best_params['scale_pos_weight'] = scale_pos_weight
#
#             print("Best parameters found by Bayesian Optimization: ", best_params)
#
#             # Train the model with the best parameters on the entire training set
#             model = xgb.XGBClassifier(**best_params)
#             model.fit(X_train, y_train)
#         else:
#             # Train with default parameters if tuning is disabled
#             model = xgb.XGBClassifier(
#                 objective='binary:logistic',
#                 eval_metric='logloss',
#                 max_depth=4,
#                 learning_rate=0.05,
#                 n_estimators=300,
#                 scale_pos_weight=scale_pos_weight,
#                 min_child_weight=5,
#                 gamma=0.1,
#                 subsample=0.8,
#                 colsample_bytree=0.8
#             )
#             model.fit(X_train, y_train)
#
#         # Save the trained model
#         joblib.dump(model, model_path)
#         print("XGBoost model trained and saved to file!")
#
#     return model
import json

import numpy as np


def feature_elimination_by_importance(X_train, y_train, X_val, y_val):
    """
    Perform recursive feature elimination based on feature importance.
    Args:
        X_train: Training feature set.
        y_train: Training labels.
        X_val: Validation feature set.
        y_val: Validation labels.
    Returns:
        List of best features.
    """
    features = X_train.columns.tolist()
    best_score = 0
    best_features = features.copy()

    while len(features) > 1:
        print(f"Testing with {len(features)} features...")

        # Train XGBoost model with current features
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False
        )
        model.fit(X_train[features], y_train)

        # Calculate ROC AUC on validation set
        y_pred = model.predict_proba(X_val[features])[:, 1]
        score = roc_auc_score(y_val, y_pred)
        print(f"Current ROC AUC: {score:.4f}")

        # Save best score and features
        if score > best_score:
            best_score = score
            best_features = features.copy()

        # Get feature importances and remove the least important feature
        importance = model.feature_importances_
        least_important = features[np.argmin(importance)]
        print(f"Removing least important feature: {least_important}")
        features.remove(least_important)

    print(f"Best ROC AUC: {best_score:.4f}")
    print(f"Best features: {best_features}")
    return best_features


def train_xgboost(X_train, y_train, model_path="xgboost_model.joblib", params_path="best_params.json",
                  features_path="selected_features.json", tune=False, with_feature_filtered=False,
                  fine_tune=False, long_run=True, trials=1000):
    """
    Train an XGBoost model with optional hyperparameter tuning, feature selection, and long-run optimization.
    Args:
        X_train: Training feature set.
        y_train: Training labels.
        model_path: Path to save the trained model.
        params_path: Path to save or load best parameters.
        features_path: Path to save or load selected features.
        tune: Whether to perform hyperparameter tuning.
        with_feature_filtered: Whether to perform feature elimination by importance.
        fine_tune: Whether to fine-tune the model on the validation set.
        long_run: Whether to perform long-running optimization using Optuna.
        trials: Number of trials for long-run optimization.
    Returns:
        Trained XGBoost model.
    """

    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

    # Split training data into training and validation sets
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    if long_run:
        print("Starting long-run optimization with Optuna...")

        def long_run_objective(trial):
            param = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'scale_pos_weight': scale_pos_weight,
                'lambda': trial.suggest_float('lambda', 1e-5, 10.0, log=True),  # L2 regularization
                'alpha': trial.suggest_float('alpha', 1e-5, 10.0, log=True)    # L1 regularization
            }

            model = xgb.XGBClassifier(**param)
            cv_scores = cross_val_score(model, X_train_split, y_train_split, scoring='roc_auc', cv=3)
            return cv_scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(long_run_objective, n_trials=trials)

        best_params = study.best_params
        best_params['objective'] = 'binary:logistic'
        best_params['eval_metric'] = 'logloss'
        best_params['scale_pos_weight'] = scale_pos_weight

        print("Best parameters found: ", best_params)

        # Save the best parameters to a JSON file
        with open(params_path, 'w') as f:
            json.dump(best_params, f)
        print(f"Optimized parameters saved to {params_path}")

    else:
        # Load previous best parameters if available
        if os.path.exists(params_path):
            with open(params_path, 'r') as file:
                best_params = json.load(file)
            print("Loaded previous best parameters: ", best_params)
        else:
            best_params = {
                'max_depth': 4,
                'learning_rate': 0.048382856372731146,
                'n_estimators': 477,
                'min_child_weight': 6,
                'gamma': 0.0001605071172415074,
                'subsample': 0.7501140945156216,
                'colsample_bytree': 0.7481462648709857,
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'scale_pos_weight': scale_pos_weight
            }
            print("Using default parameters: ", best_params)

    if tune and not long_run:
        print("Starting hyperparameter tuning...")

        def objective(trial):
            param = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': trial.suggest_int('max_depth', 3, 5),
                'learning_rate': trial.suggest_float('learning_rate', 0.04, 0.06),
                'n_estimators': trial.suggest_int('n_estimators', 400, 500),
                'scale_pos_weight': scale_pos_weight,
                'min_child_weight': trial.suggest_int('min_child_weight', 5, 7),
                'gamma': trial.suggest_float('gamma', 0.0001, 0.0005, log=True),
                'subsample': trial.suggest_float('subsample', 0.7, 0.8),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.8),
            }

            model = xgb.XGBClassifier(**param)
            cv_scores = cross_val_score(model, X_train_split, y_train_split, scoring='roc_auc', cv=3)
            return cv_scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=600)

        best_params = study.best_params
        best_params['objective'] = 'binary:logistic'
        best_params['eval_metric'] = 'logloss'
        best_params['scale_pos_weight'] = scale_pos_weight

        print("Best parameters found: ", best_params)

        # Save the best parameters to a JSON file
        with open(params_path, 'w') as file:
            json.dump(best_params, file)
        print(f"Best parameters saved to {params_path}")

    if with_feature_filtered:
        # Perform feature elimination
        if os.path.exists(features_path):
            with open(features_path, 'r') as file:
                selected_features = json.load(file)
            print(f"Loaded selected features from file: {selected_features}")
        else:
            print("Performing feature elimination...")
            selected_features = feature_elimination_by_importance(X_train_split, y_train_split, X_val, y_val)
            with open(features_path, 'w') as file:
                json.dump(selected_features, file)
            print(f"Selected features saved to {features_path}")

        # Train the model with the best parameters and selected features
        model = FeatureFilteredXGB(selected_features=selected_features, **best_params)

    else:
        model = xgb.XGBClassifier(**best_params)

    # Fine-tune the model (if enabled)
    if fine_tune:
        print("Starting fine-tuning on the validation set...")
        model = fine_tune_xgboost(model, X_val, y_val)

    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, model_path)
    print("XGBoost model trained and saved to file!")

    return model

def fine_tune_xgboost(model, X_val, y_val):
    """
    Fine-tune an XGBoost model using early stopping on a validation set.
    """
    # Ensure the model is an XGBClassifier instance
    if not isinstance(model, xgb.XGBClassifier):
        raise ValueError("The model provided is not an instance of xgb.XGBClassifier.")

    eval_set = [(X_val, y_val)]

    try:
        # Attempt to use early stopping
        model.fit(X_val, y_val, eval_set=eval_set, early_stopping_rounds=10, verbose=True)
    except TypeError:
        print("Early stopping not supported in the current XGBoost version. Training without early stopping.")
        model.fit(X_val, y_val, eval_set=eval_set, verbose=True)

    return model



def train_logistic_regression(X_train, y_train, model_path="logistic_regression_model.joblib"):
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("Logistic Regression model loaded from file.")
    else:
        model = LogisticRegression(class_weight='balanced', max_iter=1000)
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        print("Logistic Regression model trained and saved to file!")
    return model


def train_lightgbm(X_train, y_train, model_path="lightgbm_model.joblib"):
    """
    Train a LightGBM model using the training data.
    Args:
        X_train (DataFrame): Features for training.
        y_train (Series): Target variable for training.
        model_path (str): Path to save the model file.
    Returns:
        model: Trained LightGBM model.
    """
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("LightGBM model loaded from file.")
    else:
        model = lgb.LGBMClassifier(objective='binary', metric='binary_logloss')
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        print("LightGBM model trained and saved to file!")
    return model


def tune_model(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.8, 1.0]
    }
    model = GradientBoostingClassifier()
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='recall')
    grid_search.fit(X_train, y_train)
    print(f"Best parameters found: {grid_search.best_params_}")
    return grid_search.best_estimator_


def tune_model_random(X_train, y_train):
    param_dist = {
        'n_estimators': np.arange(100, 301, 100),
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5],
        'subsample': [0.8, 1.0]
    }
    model = GradientBoostingClassifier()
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=3, cv=3,
                                       scoring='recall', random_state=42)
    random_search.fit(X_train, y_train)
    print(f"Best parameters found: {random_search.best_params_}")
    return random_search.best_estimator_
