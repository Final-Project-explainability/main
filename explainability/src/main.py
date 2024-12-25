from data_loader import load_data
import LocalExplainer
from explainability.src.ModelManager import ModelManager
from explainability.src.Models.DecisionTreeModel import DecisionTreeModel
from explainability.src.Models.LogisticRegressionModel import LogisticRegressionModel
from explainability.src.Models.XGBoostModel import XGBoostModel
from preprocessing import preprocess_data, balance_data, feature_engineering, normalize_data
from sklearn.model_selection import train_test_split
import pandas as pd
import os

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
def save_test_data(X_test, y_test, file_name=None, num_records=None):
    """
    Saves X_test and y_test into a separate file in the 'data' directory.

    :param X_test: Features of the test dataset (DataFrame or numpy array).
    :param y_test: Labels of the test dataset (Series, list, or numpy array).
    :param file_name: Name of the file to save the data. If None, saves to 'data/test_data.csv'.
    :param num_records: Number of records to save. If None, save all records (default: None).
    """
    # Ensure 'data' folder exists relative to the project root
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Set the default file name if not provided
    if file_name is None:
        file_name = os.path.join(data_dir, 'test_data.csv')
    else:
        file_name = os.path.join(data_dir, file_name)

    # Convert X_test and y_test to DataFrame if needed
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)
    if not isinstance(y_test, pd.Series):
        y_test = pd.Series(y_test, name='target')

    # Check if lengths match
    if len(X_test) != len(y_test):
        raise ValueError(f"X_test and y_test must have the same length. Got {len(X_test)} and {len(y_test)}.")

    # Combine X_test and y_test into a single DataFrame
    test_data = pd.concat([X_test, y_test], axis=1)

    # Select the specified number of records
    if num_records is not None:
        test_data = test_data.iloc[:num_records]

    # Save to file
    test_data.to_csv(file_name, index=False, encoding='utf-8')
    print(f"Test data saved to {file_name}, with {len(test_data)} records.")

# General function to train or load a model
def train_or_load_model(model_name, model_class, X_train, y_train, load_model=True):
    if load_model:
        try:
            model = ModelManager.load_model(model_name)
            print("Model loaded")
        except ValueError:
            print("No pre-trained model found. Training a new model...")
            model = model_class()
            model.train(X_train, y_train)
    else:
        model = model_class()
        model.train(X_train, y_train)
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
    # Define model mapping: maps models to their respective classes
    model_mapping = {
        'DecisionTreeClassifier': {
            'class': DecisionTreeModel,
            'normalize': True,  # Requires normalization
            'balance_data': True  # Requires data balancing
        },
        'XGBClassifier': {
            'class': XGBoostModel,
            'normalize': False,  # No normalization required
            'balance_data': False  # Handles imbalance internally
        },
        'LogisticRegression': {
            'class': LogisticRegressionModel,
            'normalize': True,  # Requires normalization
            'balance_data': False  # Requires data balancing
        }
    }

    # Get the appropriate class and settings for the selected model
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
    model = train_or_load_model(model_choice, model_info['class'], X_train, y_train, load_model)

    print("")
    # Evaluate the model
    model.evaluate_model(X_test, y_test)

    if not load_model:
        ModelManager.save_model(model)

    print("\nPerforming global explanations...")
    model.global_explain(X_train=X_train,y_train=y_train)
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
    save_test_data(X_test, y_test, 'example_test_data.csv', num_records=200)
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
