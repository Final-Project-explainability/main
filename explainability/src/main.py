import os
import joblib
import xgboost as xgb
import lightgbm as lgb  # Import LightGBM
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from data_loader import load_data
from preprocessing import preprocess_data, balance_data, feature_engineering
from model import *
from evaluate import evaluate_model
from sklearn.model_selection import train_test_split
import shap
import lime
from lime import lime_tabular
from xgboost import XGBClassifier
import matplotlib.pyplot as plt




# Dictionary for model file paths
MODEL_PATHS = {
    'default': "gradient_boosting_model.joblib",
    'tuned': "tuned_gradient_boosting_model.joblib",
    'xgboost': "xgboost_model.joblib",
    'logistic': "logistic_regression_model.joblib",
    'lightgbm': "lightgbm_model.joblib"  # Path for LightGBM model
}


# General function to train or load a model
def train_or_load_model(model_name, train_func, X_train, y_train):
    model_path = MODEL_PATHS[model_name]
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"{model_name.capitalize()} model loaded from file.")
    else:
        model = train_func(X_train, y_train)
        joblib.dump(model, model_path)
        print(f"{model_name.capitalize()} model trained and saved to file.")
    return model


# Function to analyze individual mortality risk
def analyze_individual_risk(model, X_test):
    """
    Analyze and visualize the mortality risk for a specific row in the dataset.

    Args:
        model: The trained model.
        X_test (DataFrame): The test data.
    """
    while True:
        try:
            row_num = int(input("Enter row number for analysis (enter 0 to exit): "))
            if row_num == 0:
                print("Exiting analysis.")
                break

            if row_num < 0 or row_num >= len(X_test):
                print(f"Row number {row_num} is out of range. Please enter a valid row number.")
                continue

            individual_data = X_test.iloc[[row_num]]

            # Use SHAP TreeExplainer for tree-based models and KernelExplainer otherwise
            if isinstance(model, (xgb.XGBClassifier, GradientBoostingClassifier, lgb.LGBMClassifier)):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(individual_data)
                base_value = explainer.expected_value if np.ndim(explainer.expected_value) == 0 else \
                    explainer.expected_value[0]
                print("Using TreeExplainer for tree-based model.")
            else:
                explainer = shap.KernelExplainer(model.predict_proba, X_test)
                shap_values = explainer.shap_values(individual_data)
                base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else \
                    explainer.expected_value
                print("Using KernelExplainer for non-tree-based model.")

            max_display = 10
            plt.figure(figsize=(12, max_display * 1.5), dpi=80)

            shap.waterfall_plot(
                shap.Explanation(values=shap_values[0][:max_display],
                                 base_values=base_value,
                                 data=individual_data.iloc[0][:max_display]),
                max_display=max_display
            )

            # Fine-tune layout with subplots_adjust
            plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.2)  # Adjust padding as needed
            plt.savefig("shap_waterfall_plot.png", bbox_inches="tight")  # Save with tight bounding box
            plt.show()
        except ValueError:
            print("Invalid input. Please enter a numeric row number.")


# Main function
def main(model_choice='default', balance_method=None):
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'training_v2.csv')
    data = load_data(data_path)

    if data is None:
        print("Error loading the dataset")
        return

    print("Column names in the dataset:", data.columns)

    data = feature_engineering(data)
    data = preprocess_data(data)

    if 'hospital_death' in data.columns:
        X = data.drop(columns=['hospital_death'])
        y = data['hospital_death']
    else:
        print("Target column 'hospital_death' not found.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Perform data balancing if balance_method is specified
    if balance_method:
        X_train_balanced, y_train_balanced = balance_data(X_train, y_train, method=balance_method)
        print(f"Data balanced using method: {balance_method}")
    else:
        X_train_balanced, y_train_balanced = X_train, y_train  # No balancing
        print("Data not balanced.")

    # Choose model for training or loading
    if model_choice == 'tuned':
        model = train_or_load_model('tuned', tune_model, X_train_balanced, y_train_balanced)
    elif model_choice == 'xgboost':
        model = train_or_load_model('xgboost', train_xgboost, X_train_balanced, y_train_balanced)

    elif model_choice == 'logistic':
        model = train_or_load_model('logistic', train_logistic_regression, X_train_balanced, y_train_balanced)
    elif model_choice == 'lightgbm':
        model = train_or_load_model('lightgbm', train_lightgbm, X_train_balanced, y_train_balanced)
    else:
        model = train_or_load_model('default', train_model, X_train_balanced, y_train_balanced)

    evaluate_model(model, X_test, y_test)
    explain_model_with_shap(model, X_train_balanced)
    explain_model_with_lime(model, X_train_balanced, X_test)

    # Call function to analyze mortality risk for a specific individual
    analyze_individual_risk(model, X_test)


# Functions for SHAP and LIME explanations remain the same
def explain_model_with_shap(model, X_train):
    if isinstance(model, (GradientBoostingClassifier, xgb.XGBClassifier, lgb.LGBMClassifier)):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
        print("Using TreeExplainer for tree-based model.")
    elif isinstance(model, LogisticRegression):
        explainer = shap.KernelExplainer(model.predict, X_train)
        shap_values = explainer.shap_values(X_train)
        print("Using KernelExplainer for logistic regression model.")
    else:
        explainer = shap.KernelExplainer(model.predict, X_train)
        shap_values = explainer.shap_values(X_train)
        print("Using KernelExplainer for general model.")
    shap.summary_plot(shap_values, X_train)


def explain_model_with_lime(model, X_train, X_test):
    explainer = lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X_train.columns,
                                                  class_names=['Non-death', 'Death'], discretize_continuous=True)
    explanation = explainer.explain_instance(X_test.values[0], model.predict_proba)
    explanation.show_in_notebook()


if __name__ == "__main__":
    # main(model_choice='lightgbm')  # You can choose the new LightGBM model here
    main(model_choice='xgboost')
    # main(model_choice='logistic', balance_method='smote')
    # main(model_choice='logistic', balance_method='undersample')
    # main(model_choice='tuned random')
