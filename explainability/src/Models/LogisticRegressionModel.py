import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from explainability.src.Models.Model import Model


class LogisticRegressionModel(Model):
    def __init__(self):
        super().__init__()

    def train(self, X_train, y_train):
        model = LogisticRegression(class_weight='balanced', max_iter=1000)
        model.fit(X_train, y_train)

        self.model = model
        self.set_name()

        return model

    def local_explain(self, X_train, X_instance, predicted_probability):
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
        coefficients = self.model.coef_[0]
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

        explanation_df = pd.concat([top_10_features_df, other_contributions_row, contributions_sum_row],
                                   ignore_index=True)

        # Plot the feature contributions
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Contribution', y='Feature', data=explanation_df, palette='viridis')
        plt.axvline(x=0, color='grey', linestyle='--')

        # Annotate the plot with the total contribution and final prediction below the graph
        for i, row in explanation_df.iterrows():
            plt.text(row['Contribution'], i, f'{row["Contribution"]:.4f}', va='center', ha='left', fontsize=10,
                     color='black')

        # Title for the plot
        plt.title(
            "Feature Contributions for Logistic Regression Prediction.    " + f"Probability of Death: {predicted_probability:.4f} ")

        # Adjust layout to make space for the text annotations
        plt.tight_layout(rect=(0, 0, 0.9, 1))  # Adjust the layout to prevent overlap with text

        # Show the plot
        plt.show()

    def global_explain(self, X_train ,y_train):
        """
        Explain the logistic regression model using coefficients.

        Args:
            model: The trained Logistic Regression model.
            X_train: The dataset used for the model's training.

        Returns:
            coefficients: The model's coefficients for each feature.
        """
        # Get the coefficients and feature names
        coefficients = self.model.coef_[0]
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