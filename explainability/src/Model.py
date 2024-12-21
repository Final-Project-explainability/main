import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import cross_val_score
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import json
import os


random.seed(42)
np.random.seed(42)

#################################################  ONE_DECISION_TREE  #################################################
def train_pruned_decision_tree(X_train, y_train):
    """
    Train a pruned Decision Tree classifier using ccp_alpha.
    Args:
        X_train: Training feature set.
        y_train: Training labels.
    Returns:
        pruned_model: Pruned Decision Tree model.
    """
    # Train initial tree
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Get cost complexity pruning path
    path = model.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas  # List of alphas

    # Find the best alpha using cross-validation
    best_alpha = 0
    best_score = 0
    for alpha in ccp_alphas:
        temp_model = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
        temp_model.fit(X_train, y_train)
        score = cross_val_score(temp_model, X_train, y_train, cv=5, scoring='roc_auc').mean()
        if score > best_score:
            best_score = score
            best_alpha = alpha

    print(f"Best ccp_alpha: {best_alpha}")
    pruned_model = DecisionTreeClassifier(random_state=42, ccp_alpha=best_alpha)
    pruned_model.fit(X_train, y_train)
    return pruned_model


def tune_decision_tree(X_train, y_train):
    """
    Tune hyperparameters of a Decision Tree using GridSearchCV.
    Args:
        X_train: Training feature set.
        y_train: Training labels.
    Returns:
        best_model: Best Decision Tree model after tuning.
    """
    param_grid = {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }

    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_grid=param_grid,
        scoring='roc_auc',
        cv=5,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)
    best_model = grid_search.best_estimator_
    return best_model


def train_decision_tree(X_train, y_train, params_path="jsons/decision_tree_params.json"):
    """
    Train a single Decision Tree classifier with hyperparameter tuning and save the best parameters.
    Args:
        X_train: Training feature set.
        y_train: Training labels.
        params_path: Path to save or load best parameters as a JSON file.
    Returns:
        model: Trained Decision Tree model.
    """
    print("Training a single Decision Tree with hyperparameter tuning...")

    # Define hyperparameter grid
    param_grid = {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }

    # Check if we have pre-saved parameters
    if os.path.exists(params_path):
        with open(params_path, 'r') as file:
            best_params = json.load(file)
        print(f"Loaded best parameters from {params_path}: {best_params}")
        model = DecisionTreeClassifier(**best_params, random_state=42)
        model.fit(X_train, y_train)
    else:
        # Perform grid search
        grid_search = GridSearchCV(
            DecisionTreeClassifier(random_state=42),
            param_grid,
            scoring='roc_auc',
            cv=5,
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        # Get the best parameters and model
        best_params = grid_search.best_params_
        print(f"Best Parameters: {best_params}")

        # Save the best parameters to a JSON file
        os.makedirs(os.path.dirname(params_path), exist_ok=True)  # Create directory if it doesn't exist
        with open(params_path, 'w') as f:
            json.dump(best_params, f)
        print(f"Saved best parameters to {params_path}")

        model = grid_search.best_estimator_

    return model


#######################################################################################################################
def train_gradient_boosting(X_train, y_train):
    print("train gradient boosting")
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, params_path="jsons/best_params.json", fine_tune=False, tune  =False):
    """
    Train an XGBoost model with optional hyperparameter tuning, feature selection, and long-run optimization.
    Args:
        X_train: Training feature set.
        y_train: Training labels.
        params_path: Path to save or load best parameters.
        fine_tune: Whether to fine-tune the model on the validation set.
        tune: Whether to perform long-running optimization using Optuna.
    Returns:
        Trained XGBoost model.
    """
    print("Training XGBoost model...")
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

    # Split training data into training and validation sets
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    if tune:
        print("Starting long-run optimization with Optuna...")

        def long_run_objective(trial):
            param = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': trial.suggest_int('max_depth', 4, 5),  # Focusing on a narrow range for depth
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.01, log=True),  # Very low learning rate
                'n_estimators': trial.suggest_int('n_estimators', 5000, 20000),  # Allowing for very large trees
                'min_child_weight': trial.suggest_int('min_child_weight', 4, 6),
                'gamma': trial.suggest_float('gamma', 1e-8, 0.01, log=True),  # Smaller gamma values for fine-tuning
                'subsample': trial.suggest_float('subsample', 0.6, 0.8),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
                'scale_pos_weight': scale_pos_weight  # Balancing class imbalance
            }

            model = xgb.XGBClassifier(**param)
            cv_scores = cross_val_score(model, X_train_split, y_train_split, scoring='roc_auc', cv=3)
            return cv_scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(long_run_objective, n_trials=50)

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

    model = xgb.XGBClassifier(**best_params)

    # Fine-tune the model (if enabled)
    if fine_tune:
        print("Starting fine-tuning on the validation set...")
        model = fine_tune_xgboost(model, X_val, y_val)

    model.fit(X_train, y_train)
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


def train_logistic_regression(X_train, y_train):

    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train):
    """
    Train a LightGBM model using the training data.
    Args:
        X_train (DataFrame): Features for training.
        y_train (Series): Target variable for training.
        model_path (str): Path to save the model file.
    Returns:
        model: Trained LightGBM model.
    """

    model = lgb.LGBMClassifier(objective='binary', metric='binary_logloss')
    model.fit(X_train, y_train)
    print("LightGBM model trained and saved to file!")
    return model


# def tune_model(X_train, y_train):
#     param_grid = {
#         'n_estimators': [100, 200],
#         'learning_rate': [0.01, 0.1],
#         'max_depth': [3, 5],
#         'subsample': [0.8, 1.0]
#     }
#     model = GradientBoostingClassifier()
#     grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='recall')
#     grid_search.fit(X_train, y_train)
#     print(f"Best parameters found: {grid_search.best_params_}")
#     return grid_search.best_estimator_
#
#
# def tune_model_random(X_train, y_train):
#     param_dist = {
#         'n_estimators': np.arange(100, 301, 100),
#         'learning_rate': [0.01, 0.05, 0.1],
#         'max_depth': [3, 4, 5],
#         'subsample': [0.8, 1.0]
#     }
#     model = GradientBoostingClassifier()
#     random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=3, cv=3,
#                                        scoring='recall', random_state=42)
#     random_search.fit(X_train, y_train)
#     print(f"Best parameters found: {random_search.best_params_}")
#     return random_search.best_estimator_
