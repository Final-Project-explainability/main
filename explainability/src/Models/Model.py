from lime.lime_tabular import LimeTabularExplainer
import lime
import lime.lime_tabular
import datetime
from abc import ABC, abstractmethod
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    roc_curve,
)
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb


class Model(ABC):
    def __init__(self):
        self.model = None
        self.name = None

    @abstractmethod
    def backend_inherent(self, X_instance):
        pass

    def backend_local_shap(self, X_instance):
        """
        Explains a prediction using SHAP for different model types.
        Displays the 10 most important features and the sum of the remaining features in descending order of importance.

        Args:
            X_instance: A pandas DataFrame row representing the instance to explain.

        Returns:
            feature_contributions: DataFrame with features, their SHAP contributions, and absolute contributions.
        """
        # Select the appropriate SHAP explainer based on the model type
        if isinstance(self.model, xgb.XGBClassifier):
            explainer = shap.TreeExplainer(self.model)
        elif isinstance(self.model, DecisionTreeClassifier):
            explainer = shap.TreeExplainer(self.model)
        elif isinstance(self.model, LogisticRegression):
            explainer = shap.KernelExplainer(self.model.predict_proba, X_instance)
        else:
            raise ValueError(f"Model type {self.name} is not supported for SHAP explanation.")

        # Compute SHAP values
        shap_values = explainer.shap_values(X_instance)

        # Handle 3D arrays (e.g., for multiclass classification)
        if shap_values.ndim == 3:
            # Extract the contributions for the desired class (e.g., class 1)
            shap_values_class_1 = shap_values[0, :, 1]  # Assuming you want class 1 contributions
        elif isinstance(shap_values, list):
            shap_values_class_1 = shap_values[1]
        else:
            shap_values_class_1 = shap_values

        # Combine feature names and SHAP values into a DataFrame
        feature_contributions = pd.DataFrame({
            "Feature": X_instance.columns,
            "Contribution": shap_values_class_1,
            "Absolute Contribution": np.abs(shap_values_class_1)
        })

        # Sort features by absolute contribution to highlight the most impactful ones
        feature_contributions = feature_contributions.sort_values(by="Absolute Contribution", ascending=False)

        return feature_contributions

    def backend_local_lime(self, X_train, X_instance):
        """
        Explains a prediction for an XGBoost or LGBM model using LIME.
        Always saves and displays the explanation as an HTML file and an image (bar plot).

        Args:
            model: The trained model (e.g., XGBoost or LightGBM).
            X_train: A pandas DataFrame representing the training data.
            X_instance: A pandas DataFrame row representing the instance to explain.

        Returns:
            explanation_list: List of feature contributions (weights).
        """
        # Set default class names
        class_names = ['Survive', 'Death']

        # Prepare the feature names
        feature_names = X_train.columns.tolist()

        # Create LIME explainer with normalized data
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,  # training data for LIME
            feature_names=feature_names,  # Feature names
            class_names=class_names,  # Class names for the output
            mode='classification',  # Model type
            discretize_continuous=False,  # Discretize continuous features
            kernel_width=5
        )

        # Explain the single instance (normalized)
        explanation = explainer.explain_instance(
            X_instance.values[0],  # instance to explain
            self.model.predict_proba,  # Prediction function
            num_samples=1000,
            num_features=183  # להציג את כל הפיצ'רים
        )

        # Extract the intercept and explanation list
        explanation_list = explanation.as_list()

        return explanation_list

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def local_explain(self, X_train, X_instance, predicted_probability):
        pass

    @abstractmethod
    def global_explain(self, X_train ,y_train):
        pass

    @abstractmethod
    def global_explain(self, X_train):
        pass

    def set_name(self):
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.name = f"{type(self.model).__name__}_{current_datetime}"

    def get_name(self):
        return self.name

    def get_type(self):
        return type(self.model).__name__

    def predict_proba(self, individual_data):
        return self.model.predict_proba(individual_data)

    def evaluate_model(self, X_test, y_test, optimal_threshold=0.5):
        # Model predictions with probability
        y_proba = self.model.predict_proba(X_test)[:, 1]

        # Apply custom threshold for classification
        y_pred = (y_proba >= optimal_threshold).astype(int)

        # Calculate basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")

        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Calculate metrics for Top 2%, 5%, and 10% risk predictions
        top_2_percent_threshold = np.percentile(y_proba, 98)
        top_5_percent_threshold = np.percentile(y_proba, 95)
        top_10_percent_threshold = np.percentile(y_proba, 90)

        # Predictions for the highest risk categories
        top_2_preds = (y_proba >= top_2_percent_threshold).astype(int)
        top_5_preds = (y_proba >= top_5_percent_threshold).astype(int)
        top_10_preds = (y_proba >= top_10_percent_threshold).astype(int)

        top_2_recall = recall_score(y_test, top_2_preds)
        top_5_recall = recall_score(y_test, top_5_preds)
        top_10_recall = recall_score(y_test, top_10_preds)

        print(f"Recall for Top 2%: {top_2_recall:.4f}")
        print(f"Recall for Top 5%: {top_5_recall:.4f}")
        print(f"Recall for Top 10%: {top_10_recall:.4f}")

        # Plot the ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()

    def top_percent_recall(self, X_test, y_test, percentage):
        """
        Calculate recall for the top X% of predicted probabilities.

        Args:
            X_test: Test features.
            y_test: True labels for the test set.
            percentage: The top percentage of samples to calculate recall (e.g., 2 or 5).

        Returns:
            recall: The recall for the top X% predictions.
        """
        # Step 1: Predict probabilities for the test set
        probabilities = self.model.predict_proba(X_test)[:, 1]  # Probability of class 1 (hospital death)

        # Step 2: Sort the predicted probabilities in descending order
        sorted_indices = np.argsort(probabilities)[::-1]

        # Step 3: Select the top X% of the data
        top_n = int(len(probabilities) * (percentage / 100))
        top_indices = sorted_indices[:top_n]

        # Step 4: Calculate recall on the top X% predictions
        top_y_true = y_test.iloc[top_indices]
        top_y_pred = np.ones(top_n)  # Assume top predictions are all '1'

        recall = recall_score(top_y_true, top_y_pred)

        return recall
