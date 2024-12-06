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
        explain_with_shap(model, X_instance)
    elif isinstance(model, DecisionTreeClassifier):
        explain_with_decision_tree(model, X_instance)
    elif isinstance(model, LogisticRegression):
        explain_with_logistic_regression(model, X_instance, prob_death)
    else:
        print("Model type is not supported for explanation.")


def explain_with_shap(model, X_instance):
    """
    Explains a prediction using SHAP for complex models like XGBoost, LightGBM, and Random Forest.

    Args:
        model: The trained model.
        X_instance: A pandas DataFrame row representing the instance to explain.

    Returns:
        None. Displays the SHAP explanation.
    """
    # Create SHAP explainer object
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_instance)

    # Get the base value (expected value)
    base_value = explainer.expected_value

    print("Prediction explanation using SHAP:")

    # Initialize SHAP visualization
    shap.initjs()

    # Create the SHAP explanation object for the waterfall plot
    explanation = shap.Explanation(values=shap_values[0], base_values=base_value, data=X_instance.iloc[0])

    # Create waterfall plot
    shap.waterfall_plot(explanation)

    # After creating the plot, we can adjust the layout for better readability
    plt.gcf().set_size_inches(14, 10)  # Increase figure size to fit feature names properly
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # Adjust margins if needed

    # Rotate the feature names if necessary
    plt.xticks(rotation=45, ha='right')  # Rotate feature names for better visibility

    # Show the plot
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

    # Annotate with the total contribution and prediction probability
    plt.figtext(0.15, -0.05, f"Total Contribution: {total_contribution:.4f}", fontsize=12, ha='left')
    plt.figtext(0.15, -0.1, f"Prediction (Probability of Death): {predicted_probability:.4f}", fontsize=12, ha='left')

    # Adjust layout to make space for the text annotations
    plt.tight_layout(rect=(0, 0, 0.9, 1))  # Adjust the layout to prevent overlap with text

    # Show the plot
    plt.show()


def analyze_individual_risk(model, X_test):
    """
    Analyze and visualize the mortality risk for a specific row in the dataset.

    Args:
        model: The trained model.
        X_test (DataFrame): The test data to get individual patient data.
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

            # Predict the death probability for this patient
            prob_death = model.predict_proba(individual_data)[:, 1]  # Assuming class 1 is 'death'
            print(f"Predicted mortality risk for this patient: {prob_death[0]:.4f}")

            explain_prediction(model, individual_data, prob_death[0])

        except ValueError:
            print("Invalid input. Please enter a numeric row number.")
