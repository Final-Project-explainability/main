import os
import joblib
import xgboost as xgb
import lightgbm as lgb  # Import LightGBM
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from data_loader import load_data
import GlobalExplainer
import LocalExplainer
from explainability.src.ModelManager import ModelManager
from preprocessing import preprocess_data, balance_data, feature_engineering, normalize_data
from Model import *
from evaluate import evaluate_model
from sklearn.model_selection import train_test_split


# General function to train or load a model
def train_or_load_model(model_name, train_func, X_train, y_train, load_model=True):
    if load_model:
        try:
            model = ModelManager.load_model(model_name)
            print("Model loaded")
        except ValueError:
            model = train_func(X_train, y_train)
            print(f"{model_name.capitalize()} model trained")
    else:
        model = train_func(X_train, y_train)
        print(f"{model_name.capitalize()} model trained")
    return model


def select_and_train_model(X_train, y_train, model_choice='GradientBoostingClassifier', load_model=True):
    """Select, train or load a model based on the given choice."""
    model_mapping = {
        'tuned': tune_model,
        'XGBClassifier': train_xgboost,
        'LogisticRegression': train_logistic_regression,
        'LGBMClassifier': train_lightgbm,
        'GradientBoostingClassifier': train_gradient_boosting
    }

    train_func = model_mapping.get(model_choice)
    return train_or_load_model(model_choice, train_func, X_train, y_train, load_model)


# Main function
def main(model_choice='GradientBoostingClassifier', balance_method=None, load_model=True):
    data = load_data()

    if data is None:
        print("Error loading the dataset")
        return

    print("Column names in the dataset:", data.columns)

    need_to_normalize_data = model_choice == 'LogisticRegression'
    data = feature_engineering(data)
    data = preprocess_data(data)

    if 'hospital_death' in data.columns:
        X = data.drop(columns=['hospital_death'])
        y = data['hospital_death']
    else:
        print("Target column 'hospital_death' not found.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # # Perform data balancing if balance_method is specified
    # if balance_method:
    #     X_train_balanced, y_train_balanced = balance_data(X_train, y_train, method=balance_method)
    #     print(f"Data balanced using method: {balance_method}")
    # else:
    #     X_train_balanced, y_train_balanced = X_train, y_train  # No balancing
    #     print("Data not balanced.")

    if need_to_normalize_data:
        X_train, X_test = normalize_data(X_train, X_test)

    model = select_and_train_model(X_train, y_train, model_choice, load_model)

    evaluate_model(model, X_test, y_test)

    # Functions for SHAP and LIME explanations remain the same
    GlobalExplainer.explain_model(model, X_train, X_test)

    # Call function to analyze mortality risk for a specific individual
    LocalExplainer.analyze_individual_risk(model, X_test, y_test)


if __name__ == "__main__":
    # main(model_choice='lightgbm')  # You can choose the new LightGBM model here
    main('XGBClassifier')
    # main(model_choice='logistic', balance_method='smote')
    # main(model_choice='logistic', balance_method='undersample')
    # main(model_choice='tuned random')
