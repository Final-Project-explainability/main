import shap
from sklearn.linear_model import LogisticRegressionCV

from main import *
import pandas as pd
import matplotlib.pyplot as plt
from lime import lime_tabular
from XGBoostTreeApproximator.FBT import FBT
from sklearn import tree
import graphviz


def train_and_visualize_fbt(X_train, y_train, xgb_model, max_depth=5, min_forest_size=10,
                            max_number_of_conjunctions=100, pruning_method='auc'):
    """
    Trains an interpretable FBT model based on the given XGBoost model and training data,
    and visualizes the resulting decision tree.

    Args:
        X_train (DataFrame): The training data (features only).
        y_train (Series): The training labels.
        xgb_model (xgboost.Booster): Pre-trained XGBoost model.
        max_depth (int): Maximum depth for the resulting decision tree.
        min_forest_size (int): Minimum number of trees to consider after pruning.
        max_number_of_conjunctions (int): Maximum number of rules/conjunctions to extract.
        pruning_method (str): Method for pruning the forest ('auc' or other criteria).

    Returns:
        FBT: The trained FBT model.
    """
    try:
        fbt = ModelManager.load_fbt(xgb_model)
    except:  # train fbt base on the given model
        # Combine X_train and y_train into a single DataFrame
        train_data = X_train.copy()
        train_data['hospital_death'] = y_train  # Add the label column

        # Extract feature names
        feature_cols = X_train.columns.tolist()
        label_col = 'hospital_death'  # Name of the label column

        # Prepare FBT
        fbt = FBT(max_depth=max_depth,
                  min_forest_size=min_forest_size,
                  max_number_of_conjunctions=max_number_of_conjunctions,
                  pruning_method=pruning_method)

        X_train_sample = train_data.sample(frac=0.1, random_state=42)

        # Fit the FBT model to the training data
        fbt.fit(X_train_sample, feature_cols, label_col, xgb_model)

        print("FBT model trained successfully.")

        ModelManager.save_fbt(xgb_model, fbt)

    # Visualize the tree (example code for visualization)
    try:
        print("Generating visualization...")
        print(fbt.get_decision_paths(X_train))
    except Exception as e:
        print(f"Failed to generate visualization: {e}")

    return fbt


def explain_model_with_lime(model, X_train, X_test):
    # Initialize LIME explainer
    explainer = lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns,
        class_names=['Non-death', 'Death'],
        discretize_continuous=True
    )

    # Create explanation for the first test instance
    explanation = explainer.explain_instance(X_test.iloc[0].values, model.predict_proba)

    # Convert the explanation to a DataFrame for easier manipulation
    exp_df = explanation.as_list()
    exp_df = pd.DataFrame(exp_df, columns=["Feature", "Contribution"])

    # Sort the explanation by absolute contribution and get the top features
    exp_df['Absolute Contribution'] = exp_df['Contribution'].abs()
    exp_df = exp_df.sort_values(by='Absolute Contribution', ascending=False)

    # Select top 20 features, but handle the case if there are less than 20
    top_n = min(20, len(exp_df))
    top_20_exp_df = exp_df.head(top_n)

    # Display the top features
    print(f"Top {top_n} important features for this instance:")
    print(top_20_exp_df)

    # Plot the top features
    plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
    plt.barh(top_20_exp_df['Feature'], top_20_exp_df['Contribution'], color='skyblue')
    plt.xlabel('Contribution to Prediction')
    plt.title(f'Top {top_n} Important Features (LIME Explanation)')
    plt.gca().invert_yaxis()  # Invert Y-axis to have the most important feature on top
    plt.tight_layout()  # Adjust layout to make sure everything fits
    plt.show()

    # Save the explanation to an HTML file (optional)
    html = explanation.as_html()
    with open("lime_explanation_top_20.html", "w", encoding='utf-8') as f:
        f.write(html)

    # Return the top features DataFrame
    return top_20_exp_df


def explain_logistic_regression_with_coefficients(model, X_train):
    """
    Explain the logistic regression model using coefficients.

    Args:
        model: The trained Logistic Regression model.
        X_train: The dataset used for the model's training.

    Returns:
        coefficients: The model's coefficients for each feature.
    """
    # Get the coefficients and feature names
    coefficients = model.coef_[0]
    feature_names = X_train.columns

    # Create a DataFrame to display the coefficients with feature names
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
    coef_df['Absolute Coefficient'] = coef_df['Coefficient'].abs()

    # Sort by absolute value of coefficient for better interpretability
    coef_df = coef_df.sort_values(by='Absolute Coefficient', ascending=False)

    # Select only the top 20 features
    top_20_coef_df = coef_df.head(20)

    # Plot the coefficients for the top 20 features with adjusted figsize
    plt.figure(figsize=(8, 9.5))  # Adjust this ratio as needed (e.g., same as SHAP plot)

    # Create horizontal bar plot with reversed order so that the most important feature is at the top
    plt.barh(top_20_coef_df['Feature'][::-1], top_20_coef_df['Coefficient'][::-1], color='skyblue')

    plt.xlabel('Coefficient Value')
    plt.title('Top 20 Logistic Regression Coefficients')

    # Rotate feature names for better readability
    plt.yticks(rotation=0)  # Makes sure the text is horizontal
    plt.tight_layout()  # Adjusts layout so that labels fit without being cut off

    plt.show()

    return top_20_coef_df


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
    except:
        explainer = shap.TreeExplainer(model)
        print("Using TreeExplainer for tree-based model.")
        shap_values = explainer.shap_values(X_train)
        # Save the SHAP values and the model type
        ModelManager.save_shap(model, shap_values)

    shap.summary_plot(shap_values, X_train)

    return shap_values


def explain_model(model, X_train, X_test, y_train):
    if isinstance(model, LogisticRegression):
        explain_logistic_regression_with_coefficients(model, X_train)
    elif isinstance(model, DecisionTreeClassifier):
        feature_names = X_train.columns
        explain_decision_tree_with_importances(model, feature_names)
    else:
        explain_model_with_shap(model, X_train)
        explain_model_with_lime(model, X_train, X_test)
        train_and_visualize_fbt(xgb_model=model, X_train=X_train, y_train=y_train)


def explain_decision_tree_with_importances(model, feature_names):
    """
    Display and plot the feature importances from a trained Decision Tree model.
    Args:
        model: Trained Decision Tree model.
        feature_names: List of feature names (or columns of the DataFrame).
    """
    if not hasattr(model, "feature_importances_"):
        print("The model does not have feature importances.")
        return

    # Extract feature importances
    importances = model.feature_importances_

    # Combine feature names and their importances into a DataFrame
    feature_importances = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })

    # Sort features by importance in descending order
    feature_importances = feature_importances.sort_values(by="Importance", ascending=False)

    # Select the top 20 features
    top_20_features = feature_importances.head(20)

    # Plot the feature importances as a horizontal bar chart
    plt.figure(figsize=(8, 9.5))  # Adjust this ratio as needed

    # Create horizontal bar plot with reversed order so the most important feature is at the top
    plt.barh(top_20_features['Feature'][::-1], top_20_features['Importance'][::-1], color='skyblue')

    plt.xlabel('Importance')
    plt.title('Top 20 Feature Importances')

    # Ensure labels are horizontal for better readability
    plt.yticks(rotation=0)
    plt.tight_layout()  # Ensures the labels and titles fit in the plot

    plt.show()

    return top_20_features
