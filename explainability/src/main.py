from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from data_loader import load_data
from preprocessing import preprocess_data, balance_data
from model import *
from evaluate import evaluate_model
from sklearn.model_selection import train_test_split
import shap
import lime
from lime import lime_tabular
import xgboost as xgb


def main(model_choice='default', balance_method=None):
    """
    Main function to run the model training, evaluation, and explanations.
    Args:
        model_choice (str): Choose 'default', 'tuned', 'xgboost', or 'logistic' to select the model.
        balance_method (str): Method for balancing the data, options: "smote" or "undersample".
    """
    # Step 1: Load the data
    data_path = '/Users/odedatias/Documents/הנדסת מערכות תוכנה ומידע/שנה ג/סמסטר ה/מבוא לבינה מלאכותית/עבודות/מטלה 1/explainability/data/training_v2.csv'  # Update the correct path to your dataset
    data = load_data(data_path)

    # Check if data was loaded correctly
    if data is None:
        print("Error loading the dataset")
        return

    # Print column names to identify the correct target column
    print("Column names in the dataset:", data.columns)

    # Step 2: Preprocess the data
    data = preprocess_data(data)

    # Step 3: Separate features and target variable
    if 'hospital_death' in data.columns:
        X = data.drop(columns=['hospital_death'])
        y = data['hospital_death']
    else:
        print("Target column 'hospital_death' not found. Please check the correct column name.")
        return

    # Step 4: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 5: Balance the training data (if not None)
    X_train_balanced, y_train_balanced = balance_data(X_train, y_train, method=balance_method)

    # Step 6: Train the model based on the chosen option
    if model_choice == 'tuned':
        print("Performing hyperparameter tuning...")
        model = tune_model(X_train_balanced, y_train_balanced)
    elif model_choice == 'xgboost':
        print("Training with XGBoost...")
        model = train_xgboost(X_train_balanced, y_train_balanced)
    elif model_choice == 'tuned random':
        print("Training with random hyperparameter tuning...")
        model = tune_model_random(X_train_balanced, y_train_balanced)
    elif model_choice == 'logistic':
        print("Training with Logistic Regression...")
        model = train_logistic_regression(X_train_balanced, y_train_balanced)
    else:
        print("Training with default Gradient Boosting model...")
        model = train_model(X_train_balanced, y_train_balanced)

    # Step 7: Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Step 8: Explain the model with SHAP
    explain_model_with_shap(model, X_train_balanced)

    # Step 9: Explain a specific instance with LIME
    explain_model_with_lime(model, X_train_balanced, X_test)


def explain_model_with_shap(model, X_train):
    """
    Use SHAP to explain the model predictions based on the model type.
    Args:
        model: The trained model.
        X_train (DataFrame): The training data.
    """
    # Check the model type and choose the appropriate SHAP explainer
    if isinstance(model, (GradientBoostingClassifier, xgb.XGBClassifier)):
        # Use TreeExplainer for tree-based models
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
        print("Using TreeExplainer for tree-based model.")
    elif isinstance(model, LogisticRegression):
        # Use KernelExplainer for logistic regression and other non-tree models
        explainer = shap.KernelExplainer(model.predict, X_train)
        shap_values = explainer.shap_values(X_train)
        print("Using KernelExplainer for logistic regression model.")
    else:
        # Fallback to KernelExplainer for other models
        explainer = shap.KernelExplainer(model.predict, X_train)
        shap_values = explainer.shap_values(X_train)
        print("Using KernelExplainer for general model.")

    # Plot SHAP summary plot
    shap.summary_plot(shap_values, X_train)


def explain_model_with_lime(model, X_train, X_test):
    """
    Use LIME to explain a specific instance prediction.
    Args:
        model: The trained model.
        X_train (DataFrame): The training data.
        X_test (DataFrame): The test data (for instance explanation).
    """
    explainer = lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X_train.columns, class_names=['Non-death', 'Death'], discretize_continuous=True)
    explanation = explainer.explain_instance(X_test.values[0], model.predict_proba)
    explanation.show_in_notebook()


if __name__ == "__main__":
    # Options: 'default - (Gradient Boosting)', 'tuned', 'xgboost', 'tuned random', 'logistic'
    # Options: "smote", "undersample"
    # main(model_choice='tuned random')  # You can switch 'logistic' to another model and 'smote' to 'undersample'
    # main(model_choice='tuned random', balance_method='smote')
    main(model_choice='xgboost')
