import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import numpy as np


def train_model(X_train, y_train):
    """
    Train a Gradient Boosting model using the training data.
    Args:
        X_train (DataFrame): Features for training.
        y_train (Series): Target variable for training.
    Returns:
        model: Trained Gradient Boosting model.
    """
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    print("Gradient Boosting model trained successfully!") # same as xgboost
    return model


def tune_model(X_train, y_train):
    """
    Perform hyperparameter tuning using GridSearchCV with limited parameter values.
    Args:
        X_train (DataFrame): The training data features.
        y_train (Series): The training data target variable.
    Returns:
        best_model: The model with the best parameters after tuning.
    """
    param_grid = {
        'n_estimators': [100, 200],  # Reduced values
        'learning_rate': [0.01, 0.1],  # Reduced values
        'max_depth': [3, 5],  # Reduced values
        'subsample': [0.8, 1.0]  # Keeping this range
    }

    model = GradientBoostingClassifier()
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='recall')
    grid_search.fit(X_train, y_train)

    print(f"Best parameters found: {grid_search.best_params_}")
    return grid_search.best_estimator_


def tune_model_random(X_train, y_train):
    """
    Perform hyperparameter tuning using RandomizedSearchCV for faster results.
    Args:
        X_train (DataFrame): The training data features.
        y_train (Series): The training data target variable.
    Returns:
        best_model: The model with the best parameters after tuning.
    """
    param_dist = {
        'n_estimators': np.arange(100, 301, 100),  # Random choice from this range
        'learning_rate': [0.01, 0.05, 0.1],  # Random choice from this list
        'max_depth': [3, 4, 5],  # Random choice
        'subsample': [0.8, 1.0]  # Random choice
    }

    model = GradientBoostingClassifier()
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=3, cv=3, scoring='recall', random_state=42)
    random_search.fit(X_train, y_train)

    print(f"Best parameters found: {random_search.best_params_}")
    return random_search.best_estimator_


def train_xgboost(X_train, y_train):
    """
    Train an XGBoost model using the training data.
    Args:
        X_train (DataFrame): Features for training.
        y_train (Series): Target variable for training.
    Returns:
        model: Trained XGBoost model.
    """
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)
    model.fit(X_train, y_train)
    print("XGBoost model trained successfully!")
    return model


def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression model using the training data.
    Args:
        X_train (DataFrame): Features for training.
        y_train (Series): Target variable for training.
    Returns:
        model: Trained Logistic Regression model.
    """
    # model = LogisticRegression(class_weight='balanced')
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train, y_train)
    print("Logistic Regression model trained successfully!")
    return model
