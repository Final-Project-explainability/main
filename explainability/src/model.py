import os
import joblib
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np
import lightgbm as lgb


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


def train_xgboost(X_train, y_train, model_path="xgboost_model.joblib", tune=True):
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("XGBoost model loaded from file.")
    else:
        # Set scale_pos_weight based on class imbalance
        scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

        if tune:
            # Define parameter grid for hyperparameter tuning
            param_grid = {
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'n_estimators': [100, 200, 300],
                'scale_pos_weight': [1, scale_pos_weight],
                'min_child_weight': [1, 5, 10],
                'subsample': [0.8, 1],
                'colsample_bytree': [0.8, 1]
            }

            xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
            grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='f1', cv=3, verbose=1, n_jobs=-1)
            grid_search.fit(X_train, y_train)

            # Use the best model from grid search
            model = grid_search.best_estimator_
            print("Best parameters found: ", grid_search.best_params_)
        else:
            # Train with default or manually set parameters
            model = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                max_depth=5,
                learning_rate=0.1,
                n_estimators=200,
                scale_pos_weight=scale_pos_weight,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8
            )
            model.fit(X_train, y_train)

        # Save the trained model
        joblib.dump(model, model_path)
        print("XGBoost model trained and saved to file!")

    return model


def train_lightgbm(X_train, y_train, model_path="lightgbm_model.joblib"):
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("LightGBM model loaded from file.")
    else:
        model = lgb.LGBMClassifier(objective='binary', metric='binary_logloss')
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        print("LightGBM model trained and saved to file!")
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
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=3, cv=3, scoring='recall', random_state=42)
    random_search.fit(X_train, y_train)
    print(f"Best parameters found: {random_search.best_params_}")
    return random_search.best_estimator_
