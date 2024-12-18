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


# # General function to train or load a model
# def train_or_load_model(model_name, train_func, X_train, y_train, load_model=True):
#     if load_model:
#         try:
#             model = ModelManager.load_model(model_name)
#             print("Model loaded")
#         except ValueError:
#             model = train_func(X_train, y_train)
#             print(f"{model_name.capitalize()} model trained")
#     else:
#         model = train_func(X_train, y_train)
#         print(f"{model_name.capitalize()} model trained")
#     return model
#
# def select_and_train_model(X_train, y_train, model_choice='GradientBoostingClassifier', load_model=True):
#     """Select, train or load a model based on the given choice."""
#     model_mapping = {
#         'tuned': tune_model,
#         'XGBClassifier': train_xgboost,
#         'LogisticRegression': train_logistic_regression,
#         'LGBMClassifier': train_lightgbm,
#         'GradientBoostingClassifier': train_gradient_boosting,
#         'DecisionTreeClassifier': train_decision_tree
#     }
#
#     train_func = model_mapping.get(model_choice)
#     return train_or_load_model(model_choice, train_func, X_train, y_train, load_model)
#
#
# # Main function
# def main(model_choice='GradientBoostingClassifier', balance_method=None, load_model=True):
#     data = load_data()
#
#     if data is None:
#         print("Error loading the dataset")
#         return
#
#     print("Column names in the dataset:", data.columns)
#
#     need_to_normalize_data = model_choice == 'LogisticRegression' or 'DecisionTreeClassifier'
#     data = feature_engineering(data)
#     data = preprocess_data(data) #TODO: check what it is doing in the code.
#
#     if 'hospital_death' in data.columns:
#         X = data.drop(columns=['hospital_death'])
#         y = data['hospital_death']
#     else:
#         print("Target column 'hospital_death' not found.")
#         return
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     # # Perform data balancing if balance_method is specified
#     # if balance_method:
#     #     X_train_balanced, y_train_balanced = balance_data(X_train, y_train, method=balance_method)
#     #     print(f"Data balanced using method: {balance_method}")
#     # else:
#     #     X_train_balanced, y_train_balanced = X_train, y_train  # No balancing
#     #     print("Data not balanced.")
#
#     if need_to_normalize_data:
#         X_train, X_test = normalize_data(X_train, X_test)
#
#     model = select_and_train_model(X_train, y_train, model_choice, load_model)
#
#     evaluate_model(model, X_test, y_test)
#
#     # Functions for SHAP and LIME explanations remain the same
#     GlobalExplainer.explain_model(model, X_train, X_test, y_train)
#
#     # Call function to analyze mortality risk for a specific individual
#     LocalExplainer.analyze_individual_risk(model, X_test, y_test)
#
#
# if __name__ == "__main__":
#     # main(model_choice='lightgbm')  # You can choose the new LightGBM model here
#     # main('XGBClassifier')
#     main('DecisionTreeClassifier')
#     # main(model_choice='logistic', balance_method='smote')
#     # main(model_choice='logistic', balance_method='undersample')
#     # main(model_choice='tuned random')

######  new version

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


def manage_models(X_train, y_train, X_test, y_test, model_choice):
    """
    Central function to manage training, evaluation, and explanations for different models.
    Args:
        X_train: Training feature set.
        y_train: Training labels.
        X_test: Test feature set.
        y_test: Test labels.
        model_choice: Selected model from the menu.
    """
    # Define model mapping: maps models to their training, evaluation, normalization, and balancing functions
    model_mapping = {
        'GradientBoostingClassifier': {
            'train_func': train_gradient_boosting,
            'normalize': False,  # No normalization required
            'balance_data': True  # Requires data balancing
        },
        'DecisionTreeClassifier': {
            'train_func': train_decision_tree,
            'normalize': True,  # Requires normalization
            'balance_data': True  # Requires data balancing
        },
        'XGBClassifier': {
            'train_func': train_xgboost,
            'normalize': False,  # No normalization required
            'balance_data': False  # Handles imbalance internally
        },
        'LGBMClassifier': {
            'train_func': train_lightgbm,
            'normalize': False,  # No normalization required
            'balance_data': False  # Handles imbalance internally
        },
        'LogisticRegression': {
            'train_func': train_logistic_regression,
            'normalize': True,  # Requires normalization
            'balance_data': False  # Requires data balancing
        }
    }

    # Get the appropriate functions and settings for the selected model
    model_info = model_mapping.get(model_choice)
    if not model_info:
        print(f"Model {model_choice} is not supported.")
        return

    # Prompt the user to choose whether to load a pre-trained model
    print("\nWould you like to load a pre-trained model or train a new one?")
    print("1. Load pre-trained model")
    print("2. Train a new model")
    load_model_choice = input("Enter your choice: ")

    if load_model_choice == "1":
        load_model = True
    elif load_model_choice == "2":
        load_model = False
    else:
        print("Invalid choice. Defaulting to train a new model.")
        load_model = False

    # Balance data if required
    if model_info['balance_data']:
        print("")
        print(f"Balancing dataset for {model_choice}...")
        print("Select a balancing method:")
        print("1. SMOTE")
        print("2. Undersample")
        print("3. None (Skip balancing)")
        balance_choice = input("Enter your choice: ")

        balance_methods = {
            "1": "smote",
            "2": "undersample",
            "3": None
        }
        balance_method = balance_methods.get(balance_choice)
        if balance_method is None:
            print("Skipping dataset balancing.")
            print("")
        else:
            X_train, y_train = balance_data(X_train, y_train, method=balance_method)

    # Normalize data if required
    if model_info['normalize']:
        print(f"Normalizing data for {model_choice}...")
        X_train, X_test = normalize_data(X_train, X_test)

    # Train or load the model
    train_func = model_info['train_func']
    model = train_or_load_model(model_choice, train_func, X_train, y_train, load_model)

    print("")
    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    if not load_model:
        ModelManager.save_model(model)

    # Display feature importances if the model supports it
    # if model_info['support_importances'] and hasattr(model, "feature_importances_"):
    #     feature_names = X_train.columns  # Get feature names
    #     GlobalExplainer.display_feature_importances(model, feature_names)
        # plot_feature_importances(model, feature_names)

    # # Perform global explanations (e.g., SHAP) if the model supports explainability
    # if model_info['support_explainability']:

    print("\nPerforming global explanations...")
    GlobalExplainer.explain_model(model, X_train, y_train)

    # print("\nPerforming local explanations using LIME...")

    LocalExplainer.analyze_individual_risk(model, X_test, y_test, X_train)


def main():
    """
    Main function to execute the model management workflow.
    """
    # Load and preprocess the dataset
    data = load_data()
    if data is None:
        print("Error loading the dataset")
        return

    print("Column names in the dataset:", data.columns)

    # Feature engineering and preprocessing
    data = feature_engineering(data)
    data = preprocess_data(data)

    X = data.drop(columns=['hospital_death'])
    y = data['hospital_death']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Menu for model selection
    print("\nSelect a model to train and evaluate:")
    print("1. GradientBoostingClassifier")
    print("2. DecisionTreeClassifier")
    print("3. XGBClassifier")
    print("4. LGBMClassifier")
    print("5. LogisticRegression")

    choice = input("Enter the number of your choice: ")
    model_choices = {
        "1": "GradientBoostingClassifier",
        "2": "DecisionTreeClassifier",
        "3": "XGBClassifier",
        "4": "LGBMClassifier",
        "5": "LogisticRegression"
    }

    model_choice = model_choices.get(choice)
    if not model_choice:
        print("Invalid choice. Exiting.")
        return

    # Manage the selected model
    manage_models(X_train, y_train, X_test, y_test, model_choice)


if __name__ == "__main__":
    main()
