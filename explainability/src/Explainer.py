import shap
from Model import *
from explainability.src.ModelManager import ModelManager
from main import *


def explain_model_with_shap(model, X_train):
    """
    Explain the model with SHAP values, using pre-saved values if available.
    If SHAP values are not found, compute them and save both model and SHAP.
    Args:
        model: The trained model object.
        X_train: The dataset used for SHAP computation.
    Returns:
        shap_values: The SHAP values for the model.
    """

    try:
        # Attempt to load SHAP values if they exist
        shap_values = ModelManager.load_shap(model)
        print("Loaded existing SHAP values.")
    except (FileNotFoundError, ValueError):
        if isinstance(model, (GradientBoostingClassifier, xgb.XGBClassifier, lgb.LGBMClassifier)):
            explainer = shap.TreeExplainer(model)
            print("Using TreeExplainer for tree-based model.")
        elif isinstance(model, LogisticRegression):
            X_train = shap.sample(X_train, 1000)  # Take a sample of 1000 rows
            explainer = shap.KernelExplainer(model.predict, X_train)
            print("Using KernelExplainer for logistic regression model.")
        else:
            X_train = shap.sample(X_train, 1000)  # Take a sample of 1000 rows
            explainer = shap.KernelExplainer(model.predict, X_train)
            print("Using KernelExplainer for general model.")
        shap_values = explainer.shap_values(X_train)
        # Save the SHAP values and the model type
        ModelManager.save_model_and_shap(model, shap_values)

    shap.summary_plot(shap_values, X_train)
    return shap_values


def explain_model_with_lime(model, X_train, X_test):
    explainer = lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X_train.columns,
                                                  class_names=['Non-death', 'Death'], discretize_continuous=True)
    explanation = explainer.explain_instance(X_test.values[0], model.predict_proba)
    explanation.show_in_notebook()

