import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import shap
import pandas as pd
import seaborn as sns


def explain_prediction(model, X_instance, prob_death):
    """
    Explains the prediction of the given model for a specific instance.

    Args:
        model: The trained model.
        X_instance: A pandas DataFrame row representing the instance to explain.
        prob_death: The probability of death by the given model

    Returns:
        None. Displays the explanation via SHAP or other methods.
    """
    if isinstance(model, (xgb.XGBClassifier, lgb.LGBMClassifier, RandomForestClassifier)):
        explain_with_shap(model, X_instance, prob_death)
    elif isinstance(model, DecisionTreeClassifier):
        explain_with_decision_tree(model, X_instance)
    elif isinstance(model, LogisticRegression):
        explain_with_logistic_regression(model, X_instance, prob_death)
    else:
        print("Model type is not supported for explanation.")


def explain_with_shap(model, X_instance, predicted_probability):
    """
    Explains a prediction using SHAP for complex models like XGBoost, LightGBM, and Random Forest.
    Displays the 10 most important features and the sum of the remaining features in descending order of importance.

    Args:
        model: The trained model.
        X_instance: A pandas DataFrame row representing the instance to explain.
        predicted_probability: The predicted probability from the model.

    Returns:
        None. Displays the SHAP explanation.
    """
    # Create a SHAP explainer object
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_instance)

    # Extract the base value (expected value of the model predictions)
    base_value = explainer.expected_value

    # Extract SHAP values for class 1 (assuming binary classification)
    shap_values_class_1 = shap_values[1] if isinstance(shap_values, list) else shap_values

    # Combine feature names and SHAP values into a DataFrame
    feature_contributions = pd.DataFrame({
        "Feature": X_instance.columns,
        "Contribution": shap_values_class_1[0],
        "Absolute Contribution": np.abs(shap_values_class_1[0])
    })

    # Sort features by absolute contribution to highlight the most impactful ones
    feature_contributions = feature_contributions.sort_values(by="Absolute Contribution", ascending=False)

    # Select the top 10 most important features
    top_10_features = feature_contributions.head(10)
    top_10_features = top_10_features.sort_values(by="Absolute Contribution")

    # Calculate the sum of the remaining features' contributions
    other_features_contribution = feature_contributions.iloc[10:]["Contribution"].sum()

    # Add a row representing the sum of the other features
    other_features_row = pd.DataFrame({
        "Feature": ["Other Features Contribution"],
        "Contribution": [other_features_contribution],
        "Absolute Contribution": [np.abs(other_features_contribution)]
    })

    # Combine top 10 features and the other features' contribution into a single DataFrame
    explanation_df = pd.concat([other_features_row, top_10_features], ignore_index=True)

    # Create a horizontal bar chart to visualize positive and negative contributions
    plt.figure(figsize=(10, 7))

    # Set colors: blue for positive contributions, red for negative contributions
    colors = ['#1f77b4' if x < 0 else '#d62728' for x in explanation_df["Contribution"]]

    # Plot horizontal bars
    plt.barh(explanation_df["Feature"], explanation_df["Contribution"], color=colors)

    # Add a vertical line for the base value (expected value)
    plt.axvline(x=base_value, color='gray', linestyle='--', label=f"Base Value: {base_value:.4f}")

    # Add a vertical line for the predicted probability
    plt.axvline(x=predicted_probability, color='green', linestyle='-',
                label=f"Predicted Probability: {predicted_probability:.4f}")

    # Add labels, title, and legend
    plt.xlabel("Contribution to Prediction")
    plt.title("Feature Contributions Using SHAP")
    plt.legend()

    # Annotate each bar with its contribution value
    for i, v in enumerate(explanation_df["Contribution"]):
        plt.text(v, i, f"{v:.2f}", va='center', ha='left' if v > 0 else 'right', color='black')

    # Adjust layout for better display
    plt.tight_layout()
    plt.show()


def explain_with_decision_tree(model, X_instance):
    """
    Explains a prediction for a DecisionTreeClassifier by visualizing the decision path.

    Args:
        model: The trained DecisionTreeClassifier.
        X_instance: A pandas DataFrame row representing the instance to explain.

    Returns:
        None. Displays the decision path explanation.
    """
    decision_path = model.decision_path(X_instance)
    print(f"Decision Path for the instance: {decision_path}")
    # Visualize decision path using matplotlib or other visualization libraries
    plt.figure(figsize=(10, 6))
    plt.title(f"Decision Path for the Instance")
    # Visualization code for decision path can be added here.

    plt.tight_layout(rect=(0, 0, 0.9, 1))  # Adjust the layout to prevent overlap with text
    plt.show()


# Function to explain the prediction using logistic regression
def explain_with_logistic_regression(model, X_instance, predicted_probability):
    """
    Explains a prediction for LogisticRegression by showing feature importance and coefficients.

    Args:
        model: The trained LogisticRegression.
        X_instance: A pandas DataFrame row representing the instance to explain.
        predicted_probability: The predicted probability (output of model.predict_proba).

    Returns:
        None. Displays the feature importance explanation.
    """
    # Get model coefficients and calculate feature contributions
    coefficients = model.coef_[0]
    feature_contributions = coefficients * X_instance.iloc[0]

    # Calculate the sum of all contributions
    total_contribution = np.sum(feature_contributions)

    # Create DataFrame for better visualization
    explanation_df = pd.DataFrame({
        "Feature": X_instance.columns,
        "Contribution": feature_contributions
    })

    # Sort by absolute contribution for top 10 features
    explanation_df['Absolute Contribution'] = explanation_df['Contribution'].abs()
    explanation_df = explanation_df.sort_values(by='Absolute Contribution', ascending=False)

    # Select top 10 features
    top_10_features_df = explanation_df.head(10)

    # Add the "Other Features Contribution" row at the bottom
    other_contributions = np.sum(explanation_df.iloc[10:]['Contribution'])
    other_contributions_row = pd.DataFrame({
        'Feature': ['Other Features Contribution'],
        'Contribution': [other_contributions]
    })
    contributions_sum_row = pd.DataFrame({
        'Feature': ['Contributions Sum'],
        'Contribution': [total_contribution]
    })

    explanation_df = pd.concat([top_10_features_df, other_contributions_row, contributions_sum_row], ignore_index=True)

    # Print the explanation
    print("Prediction explanation using Logistic Regression:")
    print(explanation_df)

    # Plot the feature contributions
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Contribution', y='Feature', data=explanation_df, palette='viridis')
    plt.axvline(x=0, color='grey', linestyle='--')

    # Annotate the plot with the total contribution and final prediction below the graph
    for i, row in explanation_df.iterrows():
        plt.text(row['Contribution'], i, f'{row["Contribution"]:.4f}', va='center', ha='left', fontsize=10,
                 color='black')

    # Title for the plot
    plt.title("Feature Contributions for Logistic Regression Prediction.    " + f"Probability of Death: {predicted_probability:.4f} ")

    # Adjust layout to make space for the text annotations
    plt.tight_layout(rect=(0, 0, 0.9, 1))  # Adjust the layout to prevent overlap with text

    # Show the plot
    plt.show()


def analyze_individual_risk(model, X_test, y_test):
    """
    Analyze and visualize the mortality risk for a specific row in the dataset.

    Args:
        model: The trained model.
        X_test (DataFrame): The test data to get individual patient data.
        y_test:
    """
    while True:
        try:
            row_num = int(input("Enter row number for analysis (enter -1 to exit): "))
            if row_num == -1:
                print("Exiting analysis.")
                break

            if row_num < 0 or row_num >= len(X_test):
                print(f"Row number {row_num} is out of range. Please enter a valid row number.")
                continue

            individual_data = X_test.iloc[[row_num]]

            # Predict the death probability for this patient
            prob_death = model.predict_proba(individual_data)[:, 1]  # Assuming class 1 is 'death'
            print(f"Predicted mortality risk for this patient: {prob_death[0]:.4f}")

            death = y_test.iloc[row_num]
            if death == 0:
                print("The patient was alive in the end of the hospitalization")
            else:
                print("The patient was dead in the end of the hospitalization")

            explain_prediction(model, individual_data, prob_death[0])

        except ValueError:
            print("Invalid input. Please enter a numeric row number.")
