import re

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

    def global_explain_with_shap(self, X_train):
        """
        Provides a global explanation of the trained model using SHAP values.

        Args:
            X_train (pd.DataFrame): The dataset to explain.

        Returns:
            shap_values: SHAP values object for visualization and analysis.
            summary_df: DataFrame with mean absolute SHAP values per feature.
        """
        # Create SHAP explainer
        try:
            explainer = shap.Explainer(self.model, X_train)
        except Exception:
            explainer = shap.Explainer(self.model.predict, X_train)

        shap_values = explainer(X_train)

        # Compute mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

        # Handle multi-output (e.g., DecisionTreeClassifier)
        if mean_abs_shap.ndim == 2:
            mean_abs_shap = mean_abs_shap.mean(axis=1)

        # Build summary dataframe
        summary_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Contribution': mean_abs_shap
        }).sort_values(by='Contribution', ascending=False).reset_index(drop=True)

        # Plot summary
        # shap.summary_plot(shap_values, X_train, plot_type="bar", show=True)

        return summary_df

    def global_explain_with_lime(self, X_train, X_sample):
        """
        Provides a global explanation using LIME by averaging local explanations
        over a pre-sampled subset of the training data.

        Args:
            X_train (pd.DataFrame): The full training dataset (used for LIME initialization).
            X_sample (pd.DataFrame): A sampled subset of X_train to use for the global explanation.

        Returns:
            pd.DataFrame: Global explanation with 'Feature' and 'Mean |Contribution|' columns.
        """
        feature_names = X_train.columns.tolist()
        total_contribs = {feat: [] for feat in feature_names}

        # Compute local explanations and accumulate absolute contributions
        for i in range(len(X_sample)):
            instance_df = X_sample.iloc[[i]]  # preserve as DataFrame
            local_expl = self.backend_local_lime(X_train, instance_df)

            for _, row in local_expl.iterrows():
                total_contribs[row['Feature']].append(abs(row['Contribution']))

        # Aggregate to mean absolute contributions
        global_contrib = {
            feat: np.mean(total_contribs[feat]) if total_contribs[feat] else 0.0
            for feat in feature_names
        }

        summary_df = pd.DataFrame({
            'Feature': list(global_contrib.keys()),
            'Mean |Contribution|': list(global_contrib.values())
        }).sort_values(by='Mean |Contribution|', ascending=False).reset_index(drop=True)

        return summary_df

    @abstractmethod
    def backend_inherent(self, X_instance):
        pass

    def backend_local_shap(self, X_instance, X_train):
        """
        Explains a prediction using SHAP for different model types.
        Displays each feature’s contribution to the prediction (not additive to inherent).
        This version uses background data for tree models, to ensure SHAP differs from inherent explanations.

        Args:
            X_instance (pd.DataFrame): A single-row instance to explain.
            X_train (pd.DataFrame): Training set, used for background.

        Returns:
            pd.DataFrame: Feature contributions (SHAP values).
        """
        # Determine the appropriate SHAP explainer
        if hasattr(self.model, "coef_"):  # Linear models (e.g., LogisticRegression)
            explainer = shap.Explainer(self.model, X_train, feature_names=X_instance.columns)
        else:  # Tree-based models: use background data to ensure SHAP ≠ pred_contribs
            explainer = shap.TreeExplainer(self.model, data=X_train, feature_perturbation="interventional")

        # Compute SHAP values
        shap_values = explainer.shap_values(X_instance)

        # Handle binary or multiclass case
        if isinstance(shap_values, list):
            # For binary classification models, shap_values is [class_0, class_1]
            shap_values_instance = shap_values[1][0]  # class 1, first instance
        elif shap_values.ndim == 3:
            shap_values_instance = shap_values[0, :, 1]  # class 1 for one instance
        else:
            shap_values_instance = shap_values[0]  # 1D array for binary classification

        # Build DataFrame
        feature_contributions = pd.DataFrame({
            "Feature": X_instance.columns,
            "Contribution": shap_values_instance,
            "Absolute Contribution": np.abs(shap_values_instance)
        })

        # Sort by importance (absolute value)
        feature_contributions = feature_contributions.sort_values(by="Absolute Contribution", ascending=False)
        return feature_contributions.drop(columns=["Absolute Contribution"])

    # def backend_local_lime(self, X_train, X_instance):
    #     """
    #     Explains a prediction for a model using LIME.
    #     Always saves and displays the explanation as an HTML file and an image (bar plot).
    #
    #     Args:
    #         X_train: A pandas DataFrame representing the training data.
    #         X_instance: A pandas DataFrame row representing the instance to explain.
    #
    #     Returns:
    #         explanation_df: DataFrame of feature contributions (weights).
    #     """
    #     # Set default class names
    #     class_names = ['Survive', 'Death']
    #
    #     # Prepare the feature names
    #     feature_names = X_train.columns.tolist()
    #
    #     # Create LIME explainer with normalized data
    #     explainer = lime.lime_tabular.LimeTabularExplainer(
    #         training_data=X_train.values,  # training data for LIME
    #         feature_names=feature_names,  # Feature names
    #         class_names=class_names,  # Class names for the output
    #         mode='classification',  # Model type
    #         discretize_continuous=False,  # Discretize continuous features
    #         kernel_width=5
    #     )
    #
    #     # Explain the single instance (normalized)
    #     explanation = explainer.explain_instance(
    #         X_instance.values[0],  # instance to explain
    #         self.model.predict_proba,  # Prediction function
    #         num_samples=1000,
    #         num_features=183  # Show all features
    #     )
    #
    #     # Extract the explanation as a list
    #     explanation_list = explanation.as_list()
    #
    #     # Convert the explanation list to a DataFrame
    #     explanation_df = pd.DataFrame(explanation_list, columns=['Feature', 'Contribution'])
    #
    #     # Add a column for absolute contribution
    #     explanation_df['Absolute Contribution'] = explanation_df['Contribution'].abs()
    #
    #     # Sort the DataFrame by absolute contribution
    #     explanation_df = explanation_df.sort_values(by='Absolute Contribution', ascending=False)
    #     # Remove the 'Absolute Contribution' column
    #     explanation_df = explanation_df.drop(columns=['Absolute Contribution'])
    #     return explanation_df

    import re

    def backend_local_lime(self, X_train, X_instance, threshold=0.01):
        """
        Computes a local explanation using LIME for a single instance.
        Uses two-pass logic: the first to determine important features,
        and the second for full aggregation and cleaned explanation.

        Args:
            X_train (pd.DataFrame): The training data used to initialize the LIME explainer.
            X_instance (pd.DataFrame): A single row (instance) to explain.
            threshold (float): Minimum absolute contribution to consider a feature important.

        Returns:
            pd.DataFrame: Explanation table with 'Feature' and 'Contribution' columns,
                          sorted by absolute contribution.
        """

        def extract_feature_name(cond_str):
            match = re.search(r'[a-zA-Z_][a-zA-Z0-9_]*', cond_str)
            return match.group(0) if match else cond_str

        class_names = ['Survive', 'Death']
        feature_names = X_train.columns.tolist()

        # binary_features = [
        #     'elective_surgery', 'apache_post_operative', 'arf_apache', 'gcs_unable_apache',
        #     'intubated_apache', 'ventilated_apache', 'aids', 'cirrhosis', 'diabetes_mellitus',
        #     'hepatic_failure', 'immunosuppression', 'leukemia', 'lymphoma', 'solid_tumor_with_metastasis'
        # ]

        categorical_features = ['race', 'gender', 'age', 'payer_code', 'medical_specialty', 'diag_1',
                           'diag_2', 'diag_3', 'metformin', 'repaglinide', 'nateglinide',
                           'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide',
                           'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
                           'miglitol', 'troglitazone', 'tolazamide', 'insulin',
                           'glyburide-metformin', 'glipizide-metformin',
                           'glimepiride-pioglitazone', 'metformin-rosiglitazone',
                           'metformin-pioglitazone', 'change', 'diabetesMed']

        categorical_indices = [X_train.columns.get_loc(f) for f in categorical_features]
        # categorical_names = {
        #     idx: sorted(X_train.iloc[:, idx].dropna().unique().tolist())
        #     for idx in categorical_indices
        # }

        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=feature_names,
            class_names=class_names,
            mode='classification',
            discretize_continuous=True,
            categorical_features=categorical_indices,
            kernel_width=3
        )

        # Pass 1: rough extraction of important features (no aggregation yet)
        explanation = explainer.explain_instance(
            X_instance.values[0],
            self.model.predict_proba,
            num_samples=1000,
            num_features=len(feature_names)
        )

        raw_explanation = explanation.as_list()
        important_features = [
            feat
            for feat, val in raw_explanation if abs(val) >= threshold
        ]


        # Pass 2: refined explanation with aggregation by true feature name
        explanation_refined = explainer.explain_instance(
            X_instance.values[0],
            self.model.predict_proba,
            num_samples=1000,
            num_features=len(important_features)
        )

        refined_contrib = {}
        for cond_str, val in explanation_refined.as_list():
            feat = extract_feature_name(cond_str)
            refined_contrib[feat] = refined_contrib.get(feat, 0.0) + val

        contributions = []
        for feat in feature_names:
            contrib = refined_contrib.get(feat, 0.0)
            if abs(contrib) < threshold:
                contrib = 0.0
            contributions.append((feat, contrib))

        explanation_df = pd.DataFrame(contributions, columns=['Feature', 'Contribution'])
        # explanation_df['Abs'] = explanation_df['Contribution'].abs()
        # explanation_df = explanation_df.sort_values(by='Abs', ascending=False).drop(columns='Abs').reset_index(
        #     drop=True)

        return explanation_df

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def local_explain(self, X_train, X_instance, predicted_probability):
        pass
    #
    # @abstractmethod
    # def global_explain_inherent(self, X_train, y_train):
    #     pass

    @abstractmethod
    def global_explain_inherent(self, X_train):
        pass

    def set_name(self):
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.name = f"{type(self.model).__name__}_{current_datetime}"

    def get_name(self):
        return self.name

    def get_type(self):
        return type(self.model).__name__

    @abstractmethod
    def backend_get_name(self):
        pass

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
