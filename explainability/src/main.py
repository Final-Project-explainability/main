import os
import joblib
import xgboost as xgb
import lightgbm as lgb  # Import LightGBM
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from data_loader import load_data
import Explainer
from explainability.src.ModelManager import ModelManager
from preprocessing import preprocess_data, balance_data, feature_engineering
from Model import *
from evaluate import evaluate_model
from sklearn.model_selection import train_test_split
import shap
import lime
from lime import lime_tabular
import matplotlib.pyplot as plt


# General function to train or load a model
def train_or_load_model(model_name, train_func, X_train, y_train):
    try:
        model = ModelManager.load_model(model_name)
        print("Model loaded")
    except ValueError:
        model = train_func(X_train, y_train)
        print(f"{model_name.capitalize()} model trained")
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
                base_value = explainer.expected_value[1] if isinstance(explainer.expected_value,
                                                                       (list, np.ndarray)) else \
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


def select_and_train_model(X_train, y_train, model_choice='GradientBoostingClassifier'):
    """Select, train or load a model based on the given choice."""
    model_mapping = {
        'tuned': tune_model,
        'XGBClassifier': train_xgboost,
        'LogisticRegression': train_logistic_regression,
        'LGBMClassifier': train_lightgbm,
        'GradientBoostingClassifier': train_gradient_boosting
    }

    train_func = model_mapping.get(model_choice)
    return train_or_load_model(model_choice, train_func, X_train, y_train)


# Main function
def main(model_choice='GradientBoostingClassifier', balance_method=None):
    data = load_data()

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

    model = select_and_train_model(X_train, y_train, model_choice)

    evaluate_model(model, X_test, y_test)

    # Functions for SHAP and LIME explanations remain the same
    Explainer.explain_model_with_shap(model, X_train_balanced)
    Explainer.explain_model_with_lime(model, X_train_balanced, X_test)

    # Call function to analyze mortality risk for a specific individual
    analyze_individual_risk(model, X_test)


if __name__ == "__main__":
    # main(model_choice='lightgbm')  # You can choose the new LightGBM model here
    main()
    # main(model_choice='logistic', balance_method='smote')
    # main(model_choice='logistic', balance_method='undersample')
    # main(model_choice='tuned random')
